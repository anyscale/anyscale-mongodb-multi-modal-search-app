import pandas as pd
from openai import OpenAI
from typing import Any, Optional
from pathlib import Path
import ray
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
import joblib
# from tenacity import retry, stop_after_attempt


def setup_pinecone() -> None:
    pinecone_api_key = os.environ.get(
        "PINECONE_API_KEY", "8bacffe0-949c-480e-af65-5b309b7c9fe4"
    )
    pc = Pinecone(api_key=pinecone_api_key)
    pc.create_index(
        name="mongodb-demo",
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


def convert_to_pinecone_vectors(row: dict[str, Any], col: str) -> dict[str, Any]:
    return {
        "id": row["name"],
        "values": row[col],
        "metadata": {}
    }


def upsert_pinecone(batch, namespace: str = "descriptions"):
    pinecone_api_key = os.environ.get(
        "PINECONE_API_KEY", "8bacffe0-949c-480e-af65-5b309b7c9fe4"
    )
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("mongodb-demo")
    index.upsert(
        vectors=[
            {
                "id": id_,
                "values": values,
                "metadata": metadata,
            }
            for id_, values, metadata in zip(
                batch["id"], batch["values"], batch["metadata"]
            )
        ],
        namespace=namespace,
    )
    return batch

# @retry(stop=stop_after_attempt(3))
def query_embedding(base_url: str, api_key: str, text: str) -> str:
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    response = client.embeddings.create(
        input=text,
        model="thenlper/gte-large",
    )
    return response.data[0].embedding

# @retry(stop=stop_after_attempt(3))
def query_llava(base_url: str, api_key: str, text: str, image_url: str) -> str:
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
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

    return str(response.choices[0].message.content)

# @retry(stop=stop_after_attempt(3))
def query_llama(base_url: str, api_key: str, text: str) -> str:
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
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

    return str(response.choices[0].message.content)


def generate_description(text: str, image: str) -> Optional[str]:
    try:
        out = query_llava(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key="esecret_wvfv1x446u8ifxujuqkimm7wjw",
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
            api_key="esecret_wvfv1x446u8ifxujuqkimm7wjw",
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

def generate_embedding(text: str) -> Optional[list[float]]:
    try:
        out = query_embedding(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key="esecret_wvfv1x446u8ifxujuqkimm7wjw",
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
        description_embedding = generate_embedding(description)
    else:
        category = None
        description_embedding = None

    return {
        **row,
        "description": description,
        "category": category,
        "name_embedding": name_embedding,
        "description_embedding": description_embedding,
    }

def update_embeddings(row: dict[str, Any]) -> dict[str, Any]:
    name = row["name"]
    name_embedding = generate_embedding(name)
    description = row["description"]
    description_embedding = generate_embedding(description)
    return {
        **row,
        "name_embedding": name_embedding,
        "description_embedding": description_embedding,
    }

# class Embedder:
#     def __init__(self) -> None:
#         self.model = SentenceTransformer("thenlper/gte-large", device="mps")

#     def __call__(self, batch: dict[str, np.ndarray]) -> np.ndarray:
#         texts = batch["description"].tolist()
#         batch["embedding"] = self.model.encode(texts, batch_size=len(texts))
#         return batch


if __name__ == "__main__":

    # df = (
    #     pd.read_csv("/mnt/cluster_storage/myntra202305041052.csv")
    #     .dropna(subset=["name", "purl", "img", "price", "rating"], how="any")
    #     .drop_duplicates(subset=["name"])
    #     .drop_duplicates(subset=["purl"])
    #     .sample(n=1000)
    # )


    # ds = (
    #     ray.data.from_pandas(df)
    #     .repartition(1000)
    #     .map(update_record, concurrency=24, num_cpus=0.01)
    #     .repartition(2)
    #     .write_json("/mnt/cluster_storage/data_with_ai_6")
    # )
    # setup_pinecone()

    # ds = (
    #     ray.data.read_json("/mnt/cluster_storage/data_with_ai_7")
    #     .repartition(400)
    #     .map(update_embeddings, concurrency=24, num_cpus=0.01)
    #     .repartition(1)
    #     .write_json("/mnt/cluster_storage/data_with_ai_8")
    # )


    # ds = (
    #     ray.data.read_json("/mnt/cluster_storage/data_with_ai_8")
    #     .repartition(400)
    #     .map(convert_to_pinecone_vectors, concurrency=24, num_cpus=0.01, fn_kwargs={"col": "description_embedding"})
    #     .map_batches(upsert_pinecone, concurrency=10, num_cpus=0.01)
    #     .materialize()
    # )

    ds = (
        ray.data.read_json("/mnt/cluster_storage/data_with_ai_8")
        .repartition(400)
        .map(convert_to_pinecone_vectors, concurrency=24, num_cpus=0.01, fn_kwargs={"col": "name_embedding"})
        .map_batches(upsert_pinecone, concurrency=10, num_cpus=0.01, fn_kwargs=dict(namespace="names"))
        .materialize()
    )
