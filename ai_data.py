import pandas as pd
from openai import OpenAI
from pathlib import Path
from sentence_transformers import SentenceTransformer
import ray
from openai import OpenAI
import numpy as np


def query_llava(base_url: str, api_key: str, text: str, image_url: str):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    chat_completions = client.chat.completions.create(
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

    return chat_completions


def query_llama(base_url: str, api_key: str, text: str):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    chat_completions = client.chat.completions.create(
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

    return chat_completions


def generate_description(text, image):
    out = query_llava(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key="esecret_wvfv1x446u8ifxujuqkimm7wjw",
        text=f"Generate an ecommerce product description given the image and this title: {text}.",
        image_url=image,
    )
    return out.choices[0].message.content


def generate_category(title, description):
    categories = ["Tops", "Bottoms", "Dresses", "Footwear", "Accessories"]
    categories_str = ", ".join(categories)
    out = query_llama(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key="esecret_wvfv1x446u8ifxujuqkimm7wjw",
        text=(
            f"Given the title of this product: {title} and "
            f"the description: {description}, what category does it belong to? "
            f"Chose from the following categories: {categories_str}. "
            "Return the category that best fits the product. Only return the category name and nothing else."
        ),
    )
    category = out.choices[0].message.content
    if category in categories:
        return category


def update_record(row):
    last_img = row["img"].split(";")[-1].strip()
    description = generate_description(text=row["name"], image=last_img)
    category = generate_category(title=row["name"], description=description)
    return {
        **row,
        "description": description,
        "category": category,
    }


class Embedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer("thenlper/gte-large", device="mps")

    def __call__(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        texts = batch["description"].tolist()
        batch["embedding"] = self.model.encode(texts, batch_size=len(texts))
        return batch


if __name__ == "__main__":
    # data = []
    # for row in df.itertuples():
    #     description = generate_description(row.name, row.img)
    #     category = generate_category(row.name, description)
    #     embedding = generate_embedding(row.name)
    #     row_dict = row._asdict()
    #     data.append(
    #         {
    #             **row_dict,
    #             "description": description,
    #             "category": category,
    #             "embedding": embedding,
    #         }

    #     )
    # data.to_csv("data_with_ai.csv")
    df = pd.read_csv(Path(__file__).parent / "myntra202305041052.csv", nrows=100)
    ds = (
        ray.data.from_pandas(df)
        .map(update_record, concurrency=10, num_cpus=0.01)
        # .map_batches(Embedder, batch_size=10, concurrency=4, num_cpus=1)
        .write_json("data_with_ai_2")
    )

# def compute_similarity(text1, text2): ...
