"""Data pipeline to generate embeddings and metadata for Myntra items and store in mongodb."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
import numpy as np
import os
import pymongo
import pyarrow as pa
import ray
import pandas as pd
import io
from pyarrow import csv
from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateOne
from pymongo.operations import SearchIndexModel, IndexModel
from vllm.multimodal.image import ImageFeatureData, ImagePixelData
import requests
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from ray.util.accelerators import NVIDIA_TESLA_A10G
from PIL import Image

# BERT model not supported yet
# class EmbedderVLLM:
#     def __init__(self, model: str = "thenlper/gte-large"):
#         self.model = LLM(model)

#     def __call__(self, batch: dict[str, np.ndarray], col: str):
#         batch[f"{col}_embedding"] = self.model(batch[col].tolist())
#         return batch


class EmbedderSentenceTransformer:
    def __init__(self, model: str = "thenlper/gte-large"):
        self.model = SentenceTransformer(model, device="cuda")

    def __call__(
        self, batch: dict[str, np.ndarray], cols: list[str]
    ) -> dict[str, np.ndarray]:
        for col in cols:
            batch[f"{col}_embedding"] = self.model.encode(batch[col].tolist())
        return batch


def gen_description_prompt(row: dict[str, Any]) -> dict[str, Any]:
    title = row["name"]
    row["description_prompt"] = "<image>" * 1176 + (
        f"\nUSER: Generate an ecommerce product description given the image and this title: {title}.\nASSISTANT:"
    )

    return row


def keep_valid_images(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    batch["img"] = [img for img in batch["img"] if img]
    return batch


def download_image(url: str) -> np.ndarray:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception:
        return b""


def download_images(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with ThreadPoolExecutor() as executor:
        batch["img"] = list(executor.map(download_image, batch["img"]))
    return keep_valid_images(batch)


# download the image
# pass the image to the LlaVA model
class LlaVAMistral:
    def __init__(self):
        self.llm = LLM(
            model="llava-hf/llava-v1.6-mistral-7b-hf",
            **{
                "trust_remote_code": True,
                "enable_lora": False,
                "max_num_seqs": 1,
                "max_model_len": 14224,
                "gpu_memory_utilization": 0.85,
                "image_input_type": "pixel_values",
                "image_token_id": 32000,
                "image_input_shape": "1,3,336,336",
                "image_feature_size": 1176,
                "enforce_eager": True,
            },
        )
        self.sampling_params = SamplingParams(
            n=1,
            presence_penalty=0,
            frequency_penalty=0,
            repetition_penalty=1,
            length_penalty=1,
            top_p=1,
            top_k=-1,
            temperature=0,
            use_beam_search=False,
            ignore_eos=False,
            max_tokens=2048,
            seed=None,
            detokenize=True,
        )

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        prompts = batch["description_prompt"]
        images = [Image.open(io.BytesIO(img)) for img in batch["img"]]
        responses = []
        for prompt, image in zip(prompts, images):
            resp = self.llm.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": ImagePixelData(image),
                },
                sampling_params=self.sampling_params,
            )
            responses.append(resp[0].outputs[0].text)
        batch["description"] = responses
        return batch


class MistralTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
        )

    def __call__(self, row: dict, input: str, output: str):
        row[output] = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": row[input]}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="np",
        ).squeeze()
        return row


class MistralvLLM:
    def __init__(self):
        self.llm = LLM(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            max_model_len=4096,
            skip_tokenizer_init=True,
        )
        self.sampling_params = SamplingParams(
            n=1,
            presence_penalty=0,
            frequency_penalty=0,
            repetition_penalty=1,
            length_penalty=1,
            top_p=1,
            top_k=-1,
            temperature=0,
            use_beam_search=False,
            ignore_eos=False,
            max_tokens=2048,
            seed=None,
            detokenize=False,
        )

    def __call__(self, batch: dict[str, np.ndarray], input: str, output: str):
        response = self.llm.generate(
            prompt_token_ids=batch[input].tolist(), sampling_params=self.sampling_params
        )[0]
        batch[output] = np.array(response.outputs[0].token_ids, dtype=np.int32)
        return batch


class MistralDeTokenizer:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
        )

    def __call__(self, row: dict[str, Any], key: str) -> dict[str, Any]:
        row[key] = self.tokenizer.decode(row[key], skip_special_tokens=True)
        return row


class BaseClassifierPromptConstructor:
    @property
    def max_class_len(self) -> int:
        return len(max(self.classes, key=len))

    def compute_max_length_words(
        self, max_title_len: int, max_description_len: int
    ) -> int:
        max_input = len(
            self.prompt_template.format(
                title="x" * max_title_len,
                description="x" * max_description_len,
                classes_str=", ".join(self.classes),
            )
        )
        max_output = self.max_class_len
        return max_input + max_output

    def __call__(self, row):
        classes_str = ", ".join(self.classes_str)
        title = row["name"]
        description = row["description"]
        row["prompt"] = self.prompt_template.format(
            title=title,
            description=description,
            classes_str=classes_str,
        )
        return row


class CategoryPromptConstructor(BaseClassifierPromptConstructor):
    classes: list[str] = ["Tops", "Bottoms", "Dresses", "Footwear", "Accessories"]
    prompt_template: str = (
        "Given the title of this product: {title} and "
        "the description: {description}, what category does it belong to? "
        "Chose from the following categories: {classes_str}. "
        "Return the category that best fits the product. Only return the category name and nothing else."
    )


class SeasonPromptConstructor(BaseClassifierPromptConstructor):
    classes: list[str] = ["Summer", "Winter", "Spring", "Fall"]
    prompt_template: str = (
        "Given the title of this product: {title} and "
        "the description: {description}, what season does it belong to? "
        "Chose from the following seasons: {classes_str}. "
        "Return the season that best fits the product. Only return the season name and nothing else."
    )


class ColorPromptConstructor(BaseClassifierPromptConstructor):
    classes: list[str] = [
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
    prompt_template: str = (
        "Given the title of this product: {title} and "
        "the description: {description}, what color does it belong to? "
        "Chose from the following colors: {classes_str}. "
        "Return the color that best fits the product. Only return the color name and nothing else."
    )


def not_missing_data(row: dict[str, Any]) -> bool:
    if any(v is None for v in row.values()):
        return False
    return True


def keep_first(g):
    return {k: np.array([v[0]]) for k, v in g.items()}


class MongoBulkUpdate:
    def __init__(self, db: str, collection: str) -> None:
        client = MongoClient(os.environ["DB_CONNECTION_STRING"])
        self.collection = client[db][collection]

    def __call__(self, batch_df: pd.DataFrame) -> dict[str, np.ndarray]:
        # cast embedding columns from arrays to lists
        for col in ["name_embedding", "description_embedding"]:
            batch_df[col] = batch_df[col].apply(lambda x: x.tolist())
        docs = batch_df.to_dict(orient="records")
        bulk_ops = [
            UpdateOne(filter={"_id": doc["_id"]}, update={"$set": doc}, upsert=True)
            for doc in docs
        ]
        self.collection.bulk_write(bulk_ops)
        return {}


class MongoBulkInsert:
    def __init__(self, db: str, collection: str) -> None:
        client = MongoClient(os.environ["DB_CONNECTION_STRING"])
        self.collection = client[db][collection]

    def __call__(self, batch_df: pd.DataFrame) -> dict[str, np.ndarray]:
        for col in ["name_embedding", "description_embedding"]:
            batch_df[col] = batch_df[col].apply(lambda x: x.tolist())
        docs = batch_df.to_dict(orient="records")
        self.collection.insert_many(docs)
        return {}


def setup_db():
    """
    Creates the following:

    database: "myntra"
        - collection: "myntra-items-offline" with the following indices:
            - An index on the "name" field with a standard lucene analyzer
            - A vector index on the embedding fields
            - Single field indices on the rest of the search fields
    """
    mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
    db = mongo_client["myntra"]
    db.drop_collection("myntra-items-offline")
    my_collection = db["myntra-items-offline"]

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
    my_collection = db["myntra-items-offline"]
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

    return ds_deduped.limit(nsamples)  # .random_sample(frac, seed=42)


def run_pipeline(path: str, nsamples: int):
    ray.init(
        runtime_env={
            "env_vars": {
                "ANYSCALE_API_KEY": os.environ["ANYSCALE_API_KEY"],
                "DB_CONNECTION_STRING": os.environ["DB_CONNECTION_STRING"],
            },
        }
    )
    ds = read_data(path)
    ds = preprocess_and_sample_data(ds, nsamples)

    # generate description using LLAVA model
    ds = (
        ds.repartition(nsamples)
        .map_batches(download_images, num_cpus=4)
        .map(gen_description_prompt)
        .map_batches(
            LlaVAMistral,
            batch_size=1,
            num_gpus=1,
            concurrency=1,
            accelerator_type=NVIDIA_TESLA_A10G,
        )
    )

    # # generate embeddings
    # (
    #     ds.map_batches(
    #         EmbedderSentenceTransformer,
    #         fn_kwargs={"cols": ["name", "description"]},
    #         batch_size=10,
    #         num_gpus=4 / 32,  # 4 GB / 32 GB
    #         concurrency=1,
    #         accelerator_type=NVIDIA_TESLA_A10G,
    #     )
    # )

    # # generate synthetic classes
    # for max_tokens, synthetic_col, prompt_gen_func in [
    #     ("category", generate_category_prompt, max_tokens_category_prompt),
    #     ("season", generate_season_prompt, max_tokens_season_prompt),
    #     ("color", generate_color_prompt, max_tokens_color_prompt)
    #     ]:
    #     ds = (
    #         ds
    #         .map(prompt_gen_func, fn_kwargs={"col":synthetic_col})
    #         .map(
    #             MistralTokenizer,
    #             fn_kwargs={
    #                 "input": "prompt",
    #                 "output": "prompt_tokens",
    #             },
    #             concurrency=1,
    #             num_cpus=1,
    #         )
    #         .map_batches(
    #             MistralvLLM,
    #             fn_kwargs={
    #                 "input": "prompt_tokens",
    #                 "output": "prompt_response",
    #             },
    #             fn_constructor_kwargs={
    #                 "max_model_len": max_tokens,
    #             }
    #             batch_size=1,
    #             num_gpus=1,
    #             concurrency=1,
    #             accelerator_type=NVIDIA_TESLA_A10G,
    #         )
    #         .map(
    #             MistralDeTokenizer,
    #             fn_kwargs={"key": "prompt_response"},
    #             concurrency=1,
    #             num_cpus=1,
    #         )
    #     )

    # # join the prompt responses

    # write out to db
    ds.write_parquet("/mnt/cluster_storage/offline_pipeline.parquet")


if __name__ == "__main__":
    print("Running pipeline")
    # setup_db()
    # clear_data_in_db()
    run_pipeline(
        path="s3://anyscale-public-materials/mongodb-demo/raw/myntra_subset.csv",
        nsamples=10,
    )
