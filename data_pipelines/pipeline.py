"""Data pipeline to generate embeddings and metadata for Myntra items and store in mongodb."""

from typing import Any, Optional
import numpy as np
import os
import pymongo
import pyarrow as pa
import ray
from openai import OpenAI
from pyarrow import csv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.operations import SearchIndexModel, IndexModel


def query_embedding(
    base_url: str, api_key: str, text: str, retries: int = 6
) -> list[float]:
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    while True:
        try:
            response = client.embeddings.create(
                input=text,
                model="thenlper/gte-large",
            )
        except Exception as e:
            retries -= 1
            if retries == 0:
                raise e
            continue
        break

    return response.data[0].embedding


def query_llava(
    base_url: str, api_key: str, text: str, image_url: str, retries: int = 6
) -> str:
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    while True:
        try:
            response = client.chat.completions.create(
                model="llava-hf/llava-v1.6-mistral-7b-hf",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
                temperature=0,
                stream=False,
            )
        except Exception as e:
            retries -= 1
            if retries == 0:
                raise e
            continue
        break

    return str(response.choices[0].message.content)


def query_llama(base_url: str, api_key: str, text: str, retries: int = 6) -> str:
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    while True:
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-2-70b-chat-hf",
                messages=[
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                temperature=0,
                stream=False,
            )
        except Exception as e:
            retries -= 1
            if retries == 0:
                raise e
            continue
        break

    return str(response.choices[0].message.content)


def generate_description(text: str, image: str) -> Optional[str]:
    try:
        out = query_llava(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=os.environ["ANYSCALE_API_KEY"],
            text=f"Generate an ecommerce product description given the image and this title: {text}.",
            image_url=image,
        )
    except Exception:
        out = None
    return out


def generate_category(title: str, description: str) -> Optional[str]:
    categories = ["Tops", "Bottoms", "Dresses", "Footwear", "Accessories"]
    categories_str = ", ".join(categories)
    try:
        category = query_llama(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=os.environ["ANYSCALE_API_KEY"],
            text=(
                f"Given the title of this product: {title} and "
                f"the description: {description}, what category does it belong to? "
                f"Chose from the following categories: {categories_str}. "
                "Return the category that best fits the product. Only return the category name and nothing else."
            ),
        )

    except Exception:
        return None

    if category in categories:
        return category

    return None


def generate_season(title: str, description: str) -> Optional[str]:
    seasons = ["Summer", "Winter", "Spring", "Fall"]
    seasons_str = ", ".join(seasons)
    try:
        season = query_llama(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=os.environ["ANYSCALE_API_KEY"],
            text=(
                f"Given the title of this product: {title} and "
                f"the description: {description}"
                f"Chose from the following seasons: {seasons_str}. "
                "Return the season that best fits the product. Only return the season name and nothing else."
            ),
        )
    except Exception:
        return None

    if season in seasons:
        return season

    return None


def generate_color(title: str, description: str) -> Optional[str]:
    colors = [
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
    ]
    colors_str = ", ".join(colors)
    try:
        color = query_llama(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=os.environ["ANYSCALE_API_KEY"],
            text=(
                f"Given the title of this product: {title} and "
                f"the description: {description}"
                f"Chose from the following colors: {colors_str}. "
                "Return the color that best fits the product. Only return the color name and nothing else."
            ),
        )
    except Exception:
        return None

    if color in colors:
        return color

    return None


def generate_embedding(text: str) -> Optional[list[float]]:
    try:
        out = query_embedding(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=os.environ["ANYSCALE_API_KEY"],
            text=text,
        )
    except Exception:
        out = None
    return out


def update_record(row: dict[str, Any]) -> dict[str, Any]:
    last_img = row["img"].split(";")[-1].strip()
    name = row["name"]
    name_embedding = generate_embedding(name)

    description = generate_description(text=name, image=last_img)
    if description is not None:
        category = generate_category(title=name, description=description)
        season = generate_season(title=name, description=description)
        color = generate_color(title=name, description=description)
        description_embedding = generate_embedding(description)
    else:
        category = None
        season = None
        color = None
        description_embedding = None

    return {
        "_id": name,
        "name": name,
        "img": row["img"],
        "price": row["price"],
        "rating": row["rating"],
        "description": description,
        "category": category,
        "season": season,
        "color": color,
        "name_embedding": name_embedding,
        "description_embedding": description_embedding,
    }


def not_missing_data(row: dict[str, Any]) -> bool:
    if any(v is None for v in row.values()):
        return False
    return True


def keep_first(g):
    return {k: np.array([v[0]]) for k, v in g.items()}


def setup_db():
    """
    Creates the following:

    database: "myntra"
        - collection: "myntra-items" with the following indices:
            - An index on the "name" field with a standard lucene analyzer
            - A vector index on the embedding fields
            - Single field indices on the rest of the search fields
    """
    mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
    db = mongo_client["myntra"]
    db.drop_collection("myntra-items")
    my_collection = db["myntra-items"]

    my_collection.create_indexes(
        [
            IndexModel([("rating", DESCENDING)]),
            IndexModel([("category", ASCENDING)]),
            IndexModel([("season", ASCENDING)]),
            IndexModel([("color", ASCENDING)]),
        ]
    )

    # TODO - uncomment when no longer running on m0 cluster
    # my_collection.create_search_index(
    #     {
    #         "definition": {
    #             "mappings": {
    #                 "dynamic": False,
    #                 "fields": {
    #                     "name": {
    #                         "type": "string",
    #                         "analyzer": "lucene.standard",
    #                     },
    #                 },
    #             }
    #         },
    #         "name": "lexical_text_search_index",
    #     }
    # )

    # my_collection.create_search_index(
    #     {
    #         "definition": {
    #             "mappings": {
    #                 "dynamic": False,
    #                 "fields": [
    #                     {
    #                         "numDimensions": 1024,
    #                         "similarity": "cosine",
    #                         "type": "vector",
    #                         "path": "description_embedding",
    #                     },
    #                     {
    #                         "type": "filter",
    #                         "path": "category",
    #                     },
    #                     {
    #                         "type": "filter",
    #                         "path": "season",
    #                     },
    #                     {
    #                         "type": "filter",
    #                         "path": "color",
    #                     },
    #                     {
    #                         "type": "filter",
    #                         "path": "rating",
    #                     },
    #                     {
    #                         "type": "filter",
    #                         "path": "price",
    #                     },
    #                 ],
    #             }
    #         },
    #         "name": "vector_search_index",
    #     }
    # )


def clear_data_in_db():
    mongo_client: pymongo.MongoClient = pymongo.MongoClient(
        os.environ["DB_CONNECTION_STRING"],
    )
    db = mongo_client["myntra"]
    my_collection = db["myntra-items"]
    my_collection.delete_many({})


def read_data(path: str) -> ray.data.Dataset:
    ds = ray.data.read_csv(
        path,
        parse_options=csv.ParseOptions(newlines_in_values=True),
        convert_options=csv.ConvertOptions(
            column_types={
                "id": pa.int64(),
                "name": pa.string(),
                "img": pa.string(),
                "asin": pa.string(),
                "price": pa.float64(),
                "mrp": pa.float64(),
                "rating": pa.float64(),
                "ratingTotal": pa.int64(),
                "discount": pa.int64(),
                "seller": pa.string(),
                "purl": pa.string(),
            }
        ),
    )
    return ds


def preprocess_and_sample_data(ds: ray.data.Dataset, nsamples: int) -> ray.data.Dataset:
    ds_deduped = (
        # remove rows missing values
        ds.filter(
            lambda x: all(x[k] is not None for k in ["name", "img", "price", "rating"])
        )
        # drop duplicates on name
        .groupby("name").map_groups(keep_first)
    )

    count = ds_deduped.count()
    frac = nsamples / count
    print(f"Sampling {frac=} of the data")

    return ds_deduped.random_sample(frac, seed=42)


def run_pipeline(path: str, nsamples: int):
    ds = read_data(path)
    ds = preprocess_and_sample_data(ds, nsamples)
    (
        ds.repartition(nsamples)
        .map(update_record, concurrency=29, num_cpus=0.01)
        .filter(not_missing_data, concurrency=29, num_cpus=0.01)
        .write_mongo(
            uri=os.environ["DB_CONNECTION_STRING"],
            database="myntra",
            collection="myntra-items",
        )
    )


if __name__ == "__main__":
    print("Running pipeline")
    # setup_db()
    clear_data_in_db()
    run_pipeline(
        path="s3://anyscale-public-materials/mongodb-demo/raw/myntra_subset.csv",
        nsamples=200,
    )
