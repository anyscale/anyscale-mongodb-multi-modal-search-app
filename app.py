import math
import ray
import gradio as gr
import pandas as pd
from gradio_rangeslider import RangeSlider
from pathlib import Path
import openai

# df_no_ai = pd.read_csv(Path(__file__).parent / "myntra202305041052.csv", nrows=10)
df = ray.data.read_json("data_with_ai").to_pandas()


def get_embedding(text):
    client = openai.OpenAI(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key="esecret_wvfv1x446u8ifxujuqkimm7wjw",
    )

    response = client.embeddings.create(
        input=text, model="thenlper/gte-large", 
    )
    return response.data[0].embedding


# Filter function
def filter_products(keywords_str, price, min_rating):
    min_price, max_price = price
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
    return [
        (row["img"].split(";")[-1].strip(), row["name"])
        for _, row in filtered_df.iterrows()
    ]
    

def filter_products_with_ai(
    synthetic_category, text_search, keywords_str, price, min_rating
):
    min_price, max_price = price
    # text_embedding = get_embedding(text_search)
    filtered_df = df[
        (df["price"] >= min_price)
        & (df["price"] <= max_price)
        & (df["rating"] >= min_rating)
        & (df["category"] == synthetic_category)
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
    return [
        (row["img"].split(";")[-1].strip(), row["name"])
        for _, row in filtered_df.iterrows()
    ]


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
                price_range_component = RangeSlider(
                    price_min, price_max, label="Price Range"
                )
                min_rating_component = gr.Slider(
                    rating_min, rating_max, step=0.25, label="Min Rating"
                )
                filter_button_component = gr.Button("Filter")
            with gr.Column(scale=3):
                gallery = gr.Gallery(label="Filtered Products", columns=3, height=800)
        inputs = [keywords_component, price_range_component, min_rating_component]
        filter_button_component.click(filter_products, inputs=inputs, outputs=gallery)
        iface.load(
            filter_products,
            inputs=inputs,
            outputs=gallery,
        )

    with gr.Tab(label="AI enabled search"):
        with gr.Row():
            with gr.Column(scale=1):
                synthetic_category_dropdown = gr.Dropdown(
                    ["Tops", "Bottoms", "Dresses", "Footwear", "Accessories"],
                    label="Category",
                )
                text_component = gr.Textbox(label="Text Search")
                keywords_component = gr.Textbox(label="Keywords")
                price_range_component = RangeSlider(
                    price_min, price_max, label="Price Range"
                )
                min_rating_component = gr.Slider(
                    rating_min, rating_max, step=0.25, label="Min Rating"
                )
                filter_button_component = gr.Button("Filter")
            with gr.Column(scale=3):
                gallery = gr.Gallery(label="Filtered Products", columns=3, height=800)
        inputs = [
            synthetic_category_dropdown,
            text_component,
            keywords_component,
            price_range_component,
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
iface.launch()
