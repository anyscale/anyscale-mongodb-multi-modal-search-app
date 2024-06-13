from typing import Optional
import gradio as gr
from ray.serve.gradio_integrations import GradioServer
import requests

ANYSCALE_BACKEND_SERVICE_URL = "http://localhost:8000/backend"


def filter_products_legacy(
    text_query: Optional[str], min_price: int, max_price: int, min_rating: float
) -> list[tuple[str, str]]:
    response = requests.post(
        f"{ANYSCALE_BACKEND_SERVICE_URL}/legacy_search",
        json={
            "text": text_query,
            "min_price": min_price,
            "max_price": max_price,
            "min_rating": min_rating,
        },
    )
    results = response.json()
    return prepare_output(results)


def filter_products_with_ai(
    synthetic_categories: list[str],
    text_query: str,
    min_price: int,
    max_price: int,
    min_rating: float,
):
    response = requests.post(
        f"{ANYSCALE_BACKEND_SERVICE_URL}/vector_search",
        json={
            "query": text_query,
            "synthetic_categories": synthetic_categories,
            "min_price": min_price,
            "max_price": max_price,
            "min_rating": min_rating,
        },
    )
    results = response.json()
    return prepare_output(results)


def build_interface():
    price_min = 0
    price_max = 100_000

    # Get rating range
    rating_min = 0
    rating_max = 5

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
                        label="Filtered Products",
                        columns=3,
                        height=800,
                        examples_per_page=10,
                    )
            inputs = [
                keywords_component,
                min_price_component,
                max_price_component,
                min_rating_component,
            ]
            filter_button_component.click(
                filter_products_legacy, inputs=inputs, outputs=gallery
            )
            iface.load(
                filter_products_legacy,
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
                        label="Filtered Products",
                        columns=3,
                        height=800,
                        examples_per_page=10,
                    )
            inputs = [
                synthetic_category_component,
                text_component,
                min_price_component,
                max_price_component,
                min_rating_component,
            ]
            filter_button_component.click(
                filter_products_with_ai,
                inputs=inputs,
                outputs=gallery,
            )
            iface.load(
                filter_products_with_ai,
                inputs=inputs,
                outputs=gallery,
            )

    return iface


app = GradioServer.options(ray_actor_options={"num_cpus": 1}).bind(build_interface)
