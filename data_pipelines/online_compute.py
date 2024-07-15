"""Data pipeline to generate embeddings and metadata for Myntra items and store in mongodb."""
from __future__ import annotations

from typing import Any, Literal, Optional
import numpy as np
import os
import ray
from openai import OpenAI

from data_pipelines.data import MongoBulkInsert, MongoBulkUpdate, read_data


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


def query_mistral(base_url: str, api_key: str, text: str, retries: int = 6) -> str:
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    while True:
        try:
            response = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.1",
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
            base_url=os.environ["LLAVA_SERVICE_BASE_URL"],
            api_key=os.environ["LLAVA_SERVICE_API_KEY"],
            text=f"Generate an ecommerce product description given the image and this title: {text}.",
            image_url=image,
        )
    except Exception:
        out = None
    return out


def clean_response(
    response_str: str, classes: list[str]
) -> str | None:
    matches = []
    for class_ in classes:
        if class_.lower() in response_str.lower():
            matches.append(class_)
    if len(matches) == 1:
        response = matches[0]
    else:
        response = None
    return response

def generate_category(title: str, description: str) -> Optional[str]:
    categories = ["Tops", "Bottoms", "Dresses", "Footwear", "Accessories"]
    categories_str = ", ".join(categories)
    try:
        category = query_mistral(
            base_url=os.environ["MISTRAL_SERVICE_BASE_URL"],
            api_key=os.environ["MISTRAL_SERVICE_API_KEY"],
            text=(
                f"Given the title of this product: {title} and "
                f"the description: {description}, what category does it belong to? "
                f"Chose from the following categories: {categories_str}. "
                "Return the category that best fits the product. Only return the category name and nothing else."
            ),
        )

    except Exception:
        return None

    return clean_response(category, categories)


def generate_season(title: str, description: str) -> Optional[str]:
    seasons = ["Summer", "Winter", "Spring", "Fall"]
    seasons_str = ", ".join(seasons)
    try:
        season = query_mistral(
            base_url=os.environ["MISTRAL_SERVICE_BASE_URL"],
            api_key=os.environ["MISTRAL_SERVICE_API_KEY"],
            text=(
                f"Given the title of this product: {title} and "
                f"the description: {description}"
                f"Chose from the following seasons: {seasons_str}. "
                "Return the season that best fits the product. Only return the season name and nothing else."
            ),
        )
    except Exception:
        return None

    return clean_response(season, seasons)


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
        color = query_mistral(
            base_url=os.environ["MISTRAL_SERVICE_BASE_URL"],
            api_key=os.environ["MISTRAL_SERVICE_API_KEY"],
            text=(
                f"Given the title of this product: {title} and "
                f"the description: {description}"
                f"Chose from the following colors: {colors_str}. "
                "Return the color that best fits the product. Only return the color name and nothing else."
            ),
        )
    except Exception:
        return None

    return clean_response(color, colors)


def generate_embedding(text: str) -> Optional[list[float]]:
    try:
        out = query_embedding(
            base_url=os.environ["MISTRAL_SERVICE_BASE_URL"],
            api_key=os.environ["MISTRAL_SERVICE_API_KEY"],
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


def run_pipeline(
    path: str,
    nsamples: int,
    mode: Literal["first_run", "update"],
    db_name: str,
    collection_name: str,
    max_concurrency_model: int = 29,
    max_concurrency_mongo: int = 10,
    batch_size_mongo: int = 10,
):
    ray.init(
        # add env vars
        runtime_env={
            "env_vars": {
                "ANYSCALE_MISTRAL_SERVICE_API_KEY": os.environ[
                    "ANYSCALE_MISTRAL_SERVICE_API_KEY"
                ],
                "DB_CONNECTION_STRING": os.environ["DB_CONNECTION_STRING"],
            },
        }
    )

    ds = (
        read_data(path, nsamples)
        .map(update_record, concurrency=max_concurrency_model, num_cpus=0.01)
        .filter(not_missing_data, concurrency=max_concurrency_model, num_cpus=0.01)
    )

    mongo_bulk_op: MongoBulkInsert | MongoBulkUpdate
    if mode == "first_run":
        mongo_bulk_op = MongoBulkInsert
    elif mode == "update":
        mongo_bulk_op = MongoBulkUpdate

    (
        ds.map_batches(
            mongo_bulk_op,
            fn_constructor_kwargs={"db": db_name, "collection": collection_name},
            batch_size=batch_size_mongo,
            concurrency=max_concurrency_mongo,
            num_cpus=0.1,
            batch_format="pandas",
        ).materialize()
    )

    print("Done")
