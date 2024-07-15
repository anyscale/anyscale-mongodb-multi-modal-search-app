"""Data pipeline to generate embeddings and metadata for Myntra items and store in mongodb."""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Type

import numpy as np
import requests
import ray
import torchvision
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from vllm.multimodal.image import ImagePixelData
from vllm import LLM, SamplingParams
from PIL import Image

from data import MongoBulkInsert, MongoBulkUpdate, read_data

# BERT model not supported yet in vllm
# class EmbedderVLLM:
#     def __init__(self, model: str = "thenlper/gte-large"):
#         self.model = LLM(model)

#     def __call__(self, batch: dict[str, np.ndarray], col: str):
#         batch[f"{col}_embedding"] = self.model(batch[col].tolist())
#         return batch


class EmbedderSentenceTransformer:
    def __init__(self, model: str = "thenlper/gte-large", device: str = "cuda"):
        self.model = SentenceTransformer(model, device=device)

    def __call__(
        self, batch: dict[str, np.ndarray], cols: list[str]
    ) -> dict[str, np.ndarray]:
        for col in cols:
            batch[f"{col}_embedding"] = self.model.encode(  # type: ignore
                batch[col].tolist(), batch_size=len(batch[col])
            )
        return batch


DESCRIPTION_PROMPT_TEMPLATE = "<image>" * 1176 + (
    "\nUSER: Generate an ecommerce product description given the image and this title: {title}."
    "Make sure to include information about the color of the product in the description."
    "\nASSISTANT:"
)


def gen_description_prompt(row: dict[str, Any]) -> dict[str, Any]:
    title = row["name"]
    row["description_prompt"] = DESCRIPTION_PROMPT_TEMPLATE.format(title=title)

    return row


def download_image(url: str) -> bytes:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception:
        return b""


def download_images(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with ThreadPoolExecutor() as executor:
        batch["url"] = batch["img"]
        batch["img"] = list(executor.map(download_image, batch["url"]))  # type: ignore
    return batch


class LargestCenterSquare:
    """Largest center square crop for images."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        """Crop the largest center square from an image."""
        img = Image.open(io.BytesIO(row["img"]))

        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        img = torchvision.transforms.functional.resize(
            img=img,
            size=self.size,
        )

        # Then take a center crop to a square.
        w, h = img.size
        c_top = (h - self.size) // 2
        c_left = (w - self.size) // 2
        row["img"] = torchvision.transforms.functional.crop(
            img=img,
            top=c_top,
            left=c_left,
            height=self.size,
            width=self.size,
        )

        return row


class LlaVAMistralTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
        )

    def __call__(self, batch: dict[str, np.ndarray], input: str, output: str):
        batch[output] = self.tokenizer.encode(batch[input])
        return batch


class LlaVAMistral:
    def __init__(
        self,
        max_model_len: int,
        max_num_seqs: int = 400,
        max_tokens: int = 1024,
        kv_cache_dtype: str = "fp8",
    ):
        self.llm = LLM(
            model="llava-hf/llava-v1.6-mistral-7b-hf",
            trust_remote_code=True,
            enable_lora=False,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.95,
            image_input_type="pixel_values",
            image_token_id=32000,
            image_input_shape="1,3,336,336",
            image_feature_size=1176,
            kv_cache_dtype=kv_cache_dtype,
            preemption_mode="swap",
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
            max_tokens=max_tokens,
            seed=None,
            detokenize=True,
        )

    def __call__(self, batch: dict[str, np.ndarray], col: str) -> dict[str, np.ndarray]:
        prompts = batch[col]
        images = batch["img"]
        responses = self.llm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": ImagePixelData(image),
                }
                for prompt, image in zip(prompts, images)
            ],
            sampling_params=self.sampling_params,
        )

        batch["description"] = [resp.outputs[0].text for resp in responses]  # type: ignore
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
    def __init__(
        self,
        max_model_len: int = 4096,
        max_tokens: int = 2048,
        max_num_seqs: int = 256,
        kv_cache_dtype: str = "fp8",
    ):
        self.llm = LLM(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            trust_remote_code=True,
            enable_lora=False,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.95,
            skip_tokenizer_init=True,
            kv_cache_dtype=kv_cache_dtype,
            preemption_mode="swap",
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
            max_tokens=max_tokens,
            seed=None,
            detokenize=False,
        )

    def __call__(
        self, batch: dict[str, np.ndarray], input: str, output: str
    ) -> dict[str, np.ndarray]:
        responses = self.llm.generate(
            prompt_token_ids=[ids.tolist() for ids in batch[input]],
            sampling_params=self.sampling_params,
        )
        batch[output] = [resp.outputs[0].token_ids for resp in responses]  # type: ignore
        return batch


class MistralDeTokenizer:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
        )

    def __call__(self, row: dict[str, Any], key: str) -> dict[str, Any]:
        row[key] = self.tokenizer.decode(row[key], skip_special_tokens=True)
        return row


def construct_prompt_classifier(
    row: dict[str, Any],
    prompt_template: str,
    classes: list[str],
    col: str,
) -> dict[str, Any]:
    classes_str = ", ".join(classes)
    title = row["name"]
    description = row["description"]
    row[f"{col}_prompt"] = prompt_template.format(
        title=title,
        description=description,
        classes_str=classes_str,
    )
    return row


def clean_response(
    row: dict[str, Any], response_col: str, classes: list[str]
) -> dict[str, Any]:
    response_str = row[response_col]
    matches = []
    for class_ in classes:
        if class_.lower() in response_str.lower():
            matches.append(class_)
    if len(matches) == 1:
        response = matches[0]
    else:
        response = None
    row[response_col] = response
    return row


classifiers: dict[str, Any] = {
    "category": {
        "classes": ["Tops", "Bottoms", "Dresses", "Footwear", "Accessories"],
        "prompt_template": (
            "Given the title of this product: {title} and "
            "the description: {description}, what category does it belong to? "
            "Chose from the following categories: {classes_str}. "
            "Return the category that best fits the product. Only return the category name and nothing else."
        ),
        "prompt_constructor": construct_prompt_classifier,
    },
    "season": {
        "classes": ["Summer", "Winter", "Spring", "Fall"],
        "prompt_template": (
            "Given the title of this product: {title} and "
            "the description: {description}, what season does it belong to? "
            "Chose from the following seasons: {classes_str}. "
            "Return the season that best fits the product. Only return the season name and nothing else."
        ),
        "prompt_constructor": construct_prompt_classifier,
    },
    "color": {
        "classes": [
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
        "prompt_template": (
            "Given the title of this product: {title} and "
            "the description: {description}, what color does it belong to? "
            "Chose from the following colors: {classes_str}. "
            "Return the color that best fits the product. Only return the color name and nothing else."
        ),
        "prompt_constructor": construct_prompt_classifier,
    },
}


def update_record(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    batch["_id"] = batch["name"]
    return {
        "_id": batch["_id"],
        "name": batch["name"],
        "img": batch["url"],  # Does img need to remain as a URL ?
        "price": batch["price"],
        "rating": batch["rating"],
        "description": batch["description"],
        "category": batch["category_response"],
        "season": batch["season_response"],
        "color": batch["color_response"],
        "name_embedding": batch["name_embedding"].tolist(),
        "description_embedding": batch["description_embedding"].tolist(),
    }


def compute_num_tokens(row: dict[str, Any], col: str) -> dict[str, Any]:
    row["num_tokens"] = len(row[col])
    return row


def run_pipeline(
    path: str,
    nsamples: int,
    mode: Literal["first_run", "update"],
    db_name: str,
    collection_name: str,
    num_llava_tokenizer_workers: int,
    num_llava_model_workers: int,
    llava_model_accelerator_type: str,
    llava_model_batch_size: int,
    num_mistral_tokenizer_workers_per_classifier: int,
    num_mistral_model_workers_per_classifier: int,
    num_mistral_detokenizer_workers_per_classifier: int,
    mistral_model_batch_size: int,
    mistral_model_accelerator_type: str,
    num_embedder_workers: int,
    embedding_model_batch_size: int,
    embedding_model_accelerator_type: str,
    db_update_batch_size: int,
    num_db_workers: int,
):
    # 1. Read and preprocess data
    ds = read_data(path, nsamples)

    ds = (
        ds.map_batches(download_images, num_cpus=4)
        .filter(lambda x: bool(x["img"]))
        .map(LargestCenterSquare(size=336))
        .map(gen_description_prompt)
        .materialize()
    )

    # 2. Estimate input/output token distribution for LLAVA model
    max_input_tokens = (
        ds.map(
            LlaVAMistralTokenizer,
            fn_kwargs={
                "input": "description_prompt",
                "output": "description_prompt_tokens",
            },
            concurrency=num_llava_tokenizer_workers,
            num_cpus=1,
        )
        .select_columns(["description_prompt_tokens"])
        .map(compute_num_tokens, fn_kwargs={"col": "description_prompt_tokens"})
        .max(on="num_tokens")
    )
    max_output_tokens = 256  # maximum size of desired product description
    max_model_length = max_input_tokens + max_output_tokens
    print(
        f"Description gen: {max_input_tokens=} {max_output_tokens=} {max_model_length=}"
    )

    # 3. Generate description using LLAVA model inference
    ds = ds.map_batches(
        LlaVAMistral,
        fn_constructor_kwargs={
            "max_model_len": max_model_length,
            "max_tokens": max_output_tokens,
            "max_num_seqs": 400,
        },
        fn_kwargs={"col": "description_prompt"},
        batch_size=llava_model_batch_size,
        num_gpus=1,
        concurrency=num_llava_model_workers,
        accelerator_type=llava_model_accelerator_type,
    )

    # 4. Generate classifier prompts and tokenize them
    for classifier, classifier_spec in classifiers.items():
        ds = (
            ds.map(
                classifier_spec["prompt_constructor"],
                fn_kwargs={
                    "prompt_template": classifier_spec["prompt_template"],
                    "classes": classifier_spec["classes"],
                    "col": classifier,
                },
            )
            .map(
                MistralTokenizer,
                fn_kwargs={
                    "input": f"{classifier}_prompt",
                    "output": f"{classifier}_prompt_tokens",
                },
                concurrency=num_mistral_tokenizer_workers_per_classifier,
                num_cpus=1,
            )
            .materialize()
        )

    # 5. Estimate input/output token distribution for Mistral models
    for classifier, classifier_spec in classifiers.items():
        max_output_tokens = (
            ray.data.from_items(
                [
                    {
                        "output": max(classifier_spec["classes"], key=len),
                    }
                ]
            )
            .map(
                MistralTokenizer,
                fn_kwargs={
                    "input": "output",
                    "output": "output_tokens",
                },
                concurrency=1,
                num_cpus=1,
            )
            .map(
                compute_num_tokens,
                fn_kwargs={"col": "output_tokens"},
            )
            .max(on="num_tokens")
        )
        # allow for 40 tokens of buffer to account for non-exact outputs e.g "the color is Red" instead of just "Red"
        buffer_size = 40
        classifier_spec["max_output_tokens"] = max_output_tokens + buffer_size

        max_input_tokens = (
            ds.select_columns([f"{classifier}_prompt_tokens"])
            .map(compute_num_tokens, fn_kwargs={"col": f"{classifier}_prompt_tokens"})
            .max(on="num_tokens")
        )
        max_output_tokens = classifier_spec["max_output_tokens"]
        print(f"{classifier=} {max_input_tokens=} {max_output_tokens=}")
        max_model_length = max_input_tokens + max_output_tokens
        classifier_spec["max_model_length"] = max_model_length

    # 6. Generate classifier responses using Mistral model inference
    for classifier, classifier_spec in classifiers.items():
        ds = (
            ds.map_batches(
                MistralvLLM,
                fn_kwargs={
                    "input": f"{classifier}_prompt_tokens",
                    "output": f"{classifier}_response",
                },
                fn_constructor_kwargs={
                    "max_model_len": classifier_spec["max_model_length"],
                    "max_tokens": classifier_spec["max_output_tokens"],
                },
                batch_size=mistral_model_batch_size,
                num_gpus=1,
                concurrency=num_mistral_model_workers_per_classifier,
                accelerator_type=mistral_model_accelerator_type,
            )
            .map(
                MistralDeTokenizer,
                fn_kwargs={"key": f"{classifier}_response"},
                concurrency=num_mistral_detokenizer_workers_per_classifier,
                num_cpus=1,
            )
            .map(
                clean_response,
                fn_kwargs={
                    "classes": classifier_spec["classes"],
                    "response_col": f"{classifier}_response",
                },
            )
        )

    # 7. Generate embeddings using embedding model inference
    ds = ds.map_batches(
        EmbedderSentenceTransformer,
        fn_kwargs={"cols": ["name", "description"]},
        batch_size=embedding_model_batch_size,
        num_gpus=1,
        concurrency=num_embedder_workers,
        accelerator_type=embedding_model_accelerator_type,
    )

    # 8. Bulk upsert records in MongoDB
    mongo_bulk_op: Type[MongoBulkInsert] | Type[MongoBulkUpdate]
    if mode == "first_run":
        mongo_bulk_op = MongoBulkInsert
    elif mode == "update":
        mongo_bulk_op = MongoBulkUpdate

    (
        ds.map_batches(update_record)
        .map_batches(
            mongo_bulk_op,
            fn_constructor_kwargs={
                "db": db_name,
                "collection": collection_name,
            },
            batch_size=db_update_batch_size,
            concurrency=num_db_workers,
            num_cpus=0.1,
            batch_format="pandas",
        )
        .materialize()
    )
