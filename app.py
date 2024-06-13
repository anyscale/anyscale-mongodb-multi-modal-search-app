import math
from typing import Optional
import ray
import gradio as gr
import openai
from ray.serve.gradio_integrations import GradioServer
from functools import lru_cache
import os
import pymongo
from more_itertools import chunked


def get_embedding(text):
    client = openai.OpenAI(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key="esecret_wvfv1x446u8ifxujuqkimm7wjw",
    )

    response = client.embeddings.create(
        input=text,
        model="thenlper/gte-large",
    )
    return response.data[0].embedding


def compute_embedding(text):
    retries = 3
    while retries > 0:
        try:
            embedding = get_embedding(text)
            print("got embedding")
        except Exception as e:
            print(f"Error: {e}")
            embedding = None
            retries -= 1
            continue
        else:
            break
    return embedding


@lru_cache
def read_data():
    df = ray.data.read_json(
        "s3://anyscale-public-materials/mongodb-demo/data_with_ai_v3/"
    ).to_pandas()
    # df = ray.data.read_json("/mnt/cluster_storage/data_with_ai_8").to_pandas()
    return df


def filter_products_mongo(keywords_str, min_price, max_price, min_rating, n=20):
    """Use pymongo to find records based on the provided filters."""
    client = pymongo.MongoClient(
        # os.environ["MONGODB_CONN_STR"],
        "mongodb+srv://sarieddinemarwan:yLbV9diLKku0ieIm@mongodb-anyscale-demo-m.epezhiv.mongodb.net/?retryWrites=true&w=majority&appName=mongodb-anyscale-demo-marwan",
    )
    db = client.myDatabase
    collection = db["myntra-items"]
    pipeline = []
    if keywords_str.strip():
        pipeline.append(
            {
                "$search": {
                    "index": "name-index-search",
                    "text": {
                        "query": keywords_str,
                        "path": "name",
                    },
                }
            }
        )
    pipeline.extend(
        [
            {
                "$match": {
                    "price": {"$gte": min_price, "$lte": max_price},
                    "rating": {"$gte": min_rating},
                }
            },
            {
                "$limit": n,
            },
        ]
    )
    print(f"pipeline: {pipeline}")
    records = collection.aggregate(pipeline)
    results = [
        (record["img"].split(";")[-1].strip(), record["name"]) for record in records
    ]
    return results


def filter_products_with_ai_and_mongo_and_hybrid_search(
    text_search: Optional[str],
    min_price: int,
    max_price: int,
    min_rating: float,
    synthetic_categories: list[str],
    colors: list[str],
    genders: list[str],
    seasons: list[str],
    n=20,
):
    client = pymongo.MongoClient(
        # os.environ["MONGODB_CONN_STR"],
        "mongodb+srv://sarieddinemarwan:yLbV9diLKku0ieIm@mongodb-anyscale-demo-m.epezhiv.mongodb.net/?retryWrites=true&w=majority&appName=mongodb-anyscale-demo-marwan",
    )
    db = client.myDatabase
    collection = db["myntra-items"]
    if text_search.strip():
        embedding = compute_embedding(text_search)
        vector_penalty = 1
        full_text_penalty = 10
        records = collection.aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "description_embedding",
                        "queryVector": embedding,
                        "numCandidates": 100,
                        "limit": 20,
                    }
                },
                {
                    # TODO - implement $match using pre-filters instead of a post-$search step
                    "$match": {
                        "price": {"$gte": min_price, "$lte": max_price},
                        "rating": {"$gte": min_rating},
                        "category": {"$in": synthetic_categories},
                        "color": {"$in": colors},
                        "season": {"$in": seasons},
                        "gender": {"$in": genders},
                    }
                },
                {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
                {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
                {
                    "$addFields": {
                        "vs_score": {
                            "$divide": [1.0, {"$add": ["$rank", vector_penalty, 1]}]
                        }
                    }
                },
                {
                    "$project": {
                        "vs_score": 1,
                        "_id": "$docs._id",
                        "name": "$docs.name",
                        "img": "$docs.img",
                    }
                },
                {
                    "$unionWith": {
                        "coll": "myntra-items",
                        "pipeline": [
                            {
                                "$search": {
                                    "index": "name-index-search",
                                    "text": {
                                        "query": text_search,
                                        "path": "name",
                                    },
                                    # "phrase": {"query": text_search, "path": "name"},
                                }
                            },
                            {
                                # TODO - implement $match using pre-filters instead of a post-$search step
                                "$match": {
                                    "price": {"$gte": min_price, "$lte": max_price},
                                    "rating": {"$gte": min_rating},
                                    "category": {"$in": synthetic_categories},
                                    "color": {"$in": colors},
                                    "season": {"$in": seasons},
                                    "gender": {"$in": genders},
                                }
                            },
                            {"$limit": 20},
                            {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
                            {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
                            {
                                "$addFields": {
                                    "fts_score": {
                                        "$divide": [
                                            1.0,
                                            {"$add": ["$rank", full_text_penalty, 1]},
                                        ]
                                    }
                                }
                            },
                            {
                                "$project": {
                                    "fts_score": 1,
                                    "_id": "$docs._id",
                                    "name": "$docs.name",
                                    "img": "$docs.img",
                                }
                            },
                        ],
                    }
                },
                {
                    "$group": {
                        "_id": "$name",
                        "img": {"$first": "$img"},
                        "vs_score": {"$max": "$vs_score"},
                        "fts_score": {"$max": "$fts_score"},
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "img": 1,
                        "vs_score": {"$ifNull": ["$vs_score", 0]},
                        "fts_score": {"$ifNull": ["$fts_score", 0]},
                    }
                },
                {
                    "$project": {
                        "score": {"$add": ["$fts_score", "$vs_score"]},
                        "name": "$_id",
                        "img": 1,
                        "vs_score": 1,
                        "fts_score": 1,
                    }
                },
                {"$sort": {"score": -1}},
                {"$limit": n},
            ]
        )
    else:
        records = collection.aggregate(
            [
                {
                    "$match": {
                        "price": {"$gte": min_price, "$lte": max_price},
                        "rating": {"$gte": min_rating},
                        "category": {"$in": synthetic_categories},
                        "color": {"$in": colors},
                        "season": {"$in": seasons},
                        "gender": {"$in": genders},
                    }
                },
                {"$limit": n},
            ]
        )
    results = list(records)
    print(results)
    results = [
        (record["img"].split(";")[-1].strip(), record["name"]) for record in results
    ]
    return results


def build_interface():
    df = read_data()

    # Get min, max and round off to nearest 100
    price_min = df["price"].min()
    price_min = int(price_min - (price_min % 100))
    price_max = df["price"].max()
    price_max = int(price_max + (100 - price_max % 100))

    # Get rating range
    rating_min = float(df["rating"].min())
    rating_min = int(rating_min)
    rating_max = float(df["rating"].max())
    rating_max = math.ceil(rating_max)

    print(
        f"Using price range: {price_min} - {price_max}"
        f" and rating range: {rating_min} - {rating_max}"
    )

    print(type(price_min), type(price_max), type(rating_min), type(rating_max))

    # Gradio Interface
    with gr.Blocks(
        # theme="shivi/calm_foam",
        title="Multi-modal search",
    ) as iface:
        with gr.Tab(label="Legacy Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    keywords_component = gr.Textbox(label="Keywords")
                    min_price_component = gr.Slider(
                        price_min, price_max, label="Min Price", value=price_min
                    )
                    max_price_component = gr.Slider(
                        price_min, price_max, label="Max Price", value=price_max
                    )
                    min_rating_component = gr.Slider(
                        rating_min, rating_max, step=0.25, label="Min Rating"
                    )
                    max_num_results_component = gr.Slider(
                        1, 100, step=1, label="Max Results", value=20
                    )
                    filter_button_component = gr.Button("Filter")
                with gr.Column(scale=3):
                    gallery = gr.Gallery(
                        label="Filtered Products",
                        columns=3,
                        height=800,
                    )
            inputs = [
                keywords_component,
                min_price_component,
                max_price_component,
                min_rating_component,
                max_num_results_component,
            ]
            filter_button_component.click(
                filter_products_mongo, inputs=inputs, outputs=gallery
            )
            iface.load(
                filter_products_mongo,
                inputs=inputs,
                outputs=gallery,
            )

        with gr.Tab(label="AI enabled search"):
            with gr.Row():
                with gr.Column(scale=1):
                    synthetic_category_component = gr.CheckboxGroup(
                        ["Tops", "Bottoms", "Dresses", "Footwear", "Accessories"],
                        label="Category",
                        value=[
                            "Tops",
                            "Bottoms",
                            "Dresses",
                            "Footwear",
                            "Accessories",
                        ],
                    )
                    gender_component = gr.CheckboxGroup(
                        ["Male", "Female"],
                        label="Gender",
                        value=[
                            "Male",
                            "Female",
                        ],
                    )
                    season_component = gr.CheckboxGroup(
                        ["Summer", "Winter", "Spring", "Fall"],
                        label="Season",
                        value=[
                            "Summer",
                            "Winter",
                            "Spring",
                            "Fall",
                        ],
                    )
                    color_component = gr.CheckboxGroup(
                        [
                            "Red",
                            "Blue",
                            "Green",
                            "Yellow",
                            "Black",
                            "White",
                            "Pink",
                            "Purple",
                            "Orange",
                            "Brown",
                            "Grey",
                        ],
                        label="Color",
                        value=[
                            "Red",
                            "Blue",
                            "Green",
                            "Yellow",
                            "Black",
                            "White",
                            "Pink",
                            "Purple",
                            "Orange",
                            "Brown",
                            "Grey",
                        ],
                    )
                    text_component = gr.Textbox(label="Text Search")
                    min_price_component = gr.Slider(
                        price_min, price_max, label="Min Price", value=price_min
                    )
                    max_price_component = gr.Slider(
                        price_min, price_max, label="Max Price", value=price_max
                    )

                    min_rating_component = gr.Slider(
                        rating_min, rating_max, step=0.25, label="Min Rating"
                    )
                    max_num_results_component = gr.Slider(
                        1, 100, step=1, label="Max Results", value=20
                    )
                    filter_button_component = gr.Button("Filter")
                with gr.Column(scale=3):
                    gallery = gr.Gallery(
                        label="Filtered Products",
                        columns=3,
                        height=800,
                        examples_per_page=10,
                    )
            inputs = [
                text_component,
                min_price_component,
                max_price_component,
                min_rating_component,
                synthetic_category_component,
                color_component,
                gender_component,
                season_component,
                max_num_results_component,
            ]
            filter_button_component.click(
                filter_products_with_ai_and_mongo_and_hybrid_search,
                inputs=inputs,
                outputs=gallery,
            )
            iface.load(
                filter_products_with_ai_and_mongo_and_hybrid_search,
                inputs=inputs,
                outputs=gallery,
            )

    return iface


app = GradioServer.options(ray_actor_options={"num_cpus": 1}).bind(build_interface)
