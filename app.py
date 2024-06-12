import math
import ray
import gradio as gr
import openai
from ray.serve.gradio_integrations import GradioServer
from functools import lru_cache
from pinecone.grpc import PineconeGRPC as Pinecone
import os
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

def query_vector_index(text: str, threshold: float):
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
    if embedding:
        pinecone_api_key = os.environ.get(
            "PINECONE_API_KEY", "8bacffe0-949c-480e-af65-5b309b7c9fe4"
        )
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index("mongodb-demo")
        result = index.query(
            vector=embedding,
            top_k=20,
            namespace="descriptions",
        )
        print("queried pinecone")
        matches = [
            match for match in result["matches"]
        ]
        print(f"{matches=}")
        filtered = [
            match["id"] for match in result["matches"]
            if match["score"] > threshold
        ]
        print(f"{filtered=}")
        return filtered


@lru_cache
def read_data():
    df = ray.data.read_json("s3://anyscale-public-materials/mongodb-demo/data_with_ai_v3/").to_pandas()
    # df = ray.data.read_json("/mnt/cluster_storage/data_with_ai_8").to_pandas()
    return df


# Filter function
def filter_products(keywords_str, min_price, max_price, min_rating, n=200):
    df = read_data()
    filtered_df = df[
        (df["price"] >= min_price)
        & (df["price"] <= max_price)
        & (df["rating"] >= min_rating)
    ]
    if keywords_str.strip():
        keywords = keywords_str.split(",")
        # matching any of the keywords
        # filtered_df = filtered_df[
        #     filtered_df["name"].str.contains("|".join(keywords), case=False)
        # ]
        # matching all of the keywords
        filtered_df = filtered_df[
            filtered_df["name"].apply(
                lambda name: all(
                    keyword.strip().lower() in name.lower().split()
                    for keyword in keywords
                )
            )
        ]
    results =  [
        (row["img"].split(";")[-1].strip(), row["name"])
        for _, row in filtered_df.iterrows()
    ]

    return list(chunked(results, 10))


def filter_products_with_ai(
    synthetic_categories, text_search, min_price, max_price, min_rating, n=20
):
    df = read_data()
    
    filtered_df = df[
        (df["price"] >= min_price)
        & (df["price"] <= max_price)
        & (df["rating"] >= min_rating)
        & (df["category"].isin(synthetic_categories))
    ]

    if text_search.strip():
        closest_names_based_on_text_search = query_vector_index(text_search, threshold=0.8)
        filtered_df = filtered_df[
            filtered_df["name"].isin(closest_names_based_on_text_search)
        ]


    return [
        (row["img"].split(";")[-1].strip(), row["name"])
        for _, row in filtered_df.iterrows()
    ][:n]


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
                    filter_button_component = gr.Button("Filter")
                with gr.Column(scale=3):
                    gallery = gr.Gallery(
                        label="Filtered Products", columns=3, height=800, examples_per_page=10
                    )
            inputs = [
                keywords_component,
                min_price_component,
                max_price_component,
                min_rating_component,
            ]
            filter_button_component.click(
                filter_products, inputs=inputs, outputs=gallery
            )
            iface.load(
                filter_products,
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
                    filter_button_component = gr.Button("Filter")
                with gr.Column(scale=3):
                    gallery = gr.Gallery(
                        label="Filtered Products", columns=3, height=800, examples_per_page=10
                    )
            inputs = [
                synthetic_category_component,
                text_component,
                min_price_component,
                max_price_component,
                min_rating_component,
            ]
            filter_button_component.click(
                filter_products_with_ai, inputs=inputs, outputs=gallery
            )
            iface.load(
                filter_products_with_ai,
                inputs=inputs,
                outputs=gallery,
            )

    # Launch the interface
    # iface.launch()
    # iface.queue(max_size=20)
    return iface


app = GradioServer.options(ray_actor_options={"num_cpus": 1}).bind(build_interface)
# if __name__ == "__main__":
#     iface = build_interface()
