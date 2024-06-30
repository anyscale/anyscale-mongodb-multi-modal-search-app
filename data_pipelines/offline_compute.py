"""Data pipeline to generate embeddings and metadata for Myntra items and store in mongodb."""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import numpy as np
import requests
import torchvision
from ray.util.accelerators import NVIDIA_TESLA_A10G
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from vllm.multimodal.image import ImagePixelData
from vllm import LLM, SamplingParams
from PIL import Image

from data_pipelines.data import MongoBulkInsert, MongoBulkUpdate, read_data

# BERT model not supported yet in vllm
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
            batch[f"{col}_embedding"] = self.model.encode(  # type: ignore
                batch[col].tolist(), batch_size=len(batch[col])
            )
        return batch


def gen_description_prompt(row: dict[str, Any]) -> dict[str, Any]:
    title = row["name"]
    row["description_prompt"] = "<image>" * 1176 + (
        f"\nUSER: Generate an ecommerce product description given the image and this title: {title}."
        "Make sure to include information about the color of the product in the description."
        "\nASSISTANT:"
    )

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
            **{
                "trust_remote_code": True,
                "enable_lora": False,
                "max_num_seqs": max_num_seqs,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": 0.95,
                "image_input_type": "pixel_values",
                "image_token_id": 32000,
                "image_input_shape": "1,3,336,336",
                "image_feature_size": 1176,
                "kv_cache_dtype": kv_cache_dtype,
                "preemption_mode": "swap",
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
        kv_cache_dtype: str = "fp8",
    ):
        self.llm = LLM(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            max_model_len=max_model_len,
            skip_tokenizer_init=True,
            kv_cache_dtype=kv_cache_dtype,
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


def compute_num_tokens(row: dict[str, np.ndarray], col: str) -> dict[str, np.ndarray]:
    row["num_tokens"] = len(row[col])
    return row


def estimate_workers(nsamples: int) -> tuple[int, int, int]:
    if nsamples < 1000:
        num_llava_workers = 1
        num_embedder_workers = 1
        num_mistral_workers_per_classifier = 1
    elif nsamples < 10000:
        num_llava_workers = 10
        num_embedder_workers = 2
        num_mistral_workers_per_classifier = 2
    elif nsamples < 100000:
        num_llava_workers = 20
        num_embedder_workers = 4
        num_mistral_workers_per_classifier = 4
    else:
        raise NotImplementedError("More than 100k samples not supported yet")
    return num_llava_workers, num_embedder_workers, num_mistral_workers_per_classifier


def run_pipeline(
    path: str,
    nsamples: int,
    mode: Literal["first_run", "update"],
    db_name: str,
    collection_name: str,
):
    ds = read_data(path, nsamples)

    num_llava_workers, num_embedder_workers, num_mistral_workers_per_classifier = (
        estimate_workers(nsamples)
    )

    # generate description using LLAVA model
    ds = (
        ds.map_batches(download_images, num_cpus=4)
        .filter(lambda x: bool(x["img"]))
        .map(LargestCenterSquare(size=336))
        .map(gen_description_prompt)
        .materialize()
    )

    # compute input/output distribution for LLAVA model
    # this is required to resolve the maximum model sequence length
    # this is necessary to maximize KV cache capacity.
    # The longer the sequence length, the more blocks need to be allocated
    # per sequence, which reduces the number of sequences that can fit in the cache.
    max_input_tokens = (
        ds.map(
            LlaVAMistralTokenizer,
            fn_kwargs={
                "input": "description_prompt",
                "output": "description_prompt_tokens",
            },
            concurrency=1,
            num_cpus=1,
        )
        .select_columns(["description_prompt_tokens"])
        .map(compute_num_tokens, fn_kwargs={"col": "description_prompt_tokens"})
        .max(on="num_tokens")
    )
    max_output_tokens = 256
    max_model_length = max_input_tokens + max_output_tokens
    print(
        f"Description gen: {max_input_tokens=} {max_output_tokens=} {max_model_length=}"
    )

    # generate description using LLAVA model
    ds = ds.map_batches(
        LlaVAMistral,
        fn_constructor_kwargs={
            "max_model_len": max_model_length,
            "max_tokens": max_output_tokens,
            "max_num_seqs": 400,
        },
        fn_kwargs={"col": "description_prompt"},
        batch_size=80,
        num_gpus=1,
        concurrency=num_llava_workers,
        accelerator_type=NVIDIA_TESLA_A10G,
    )

    # generate embeddings
    ds = ds.map_batches(
        EmbedderSentenceTransformer,
        fn_kwargs={"cols": ["name", "description"]},
        batch_size=80,
        num_gpus=1,
        concurrency=num_embedder_workers,
        accelerator_type=NVIDIA_TESLA_A10G,
    )

    # compute classifier outputs locally
    # TODO - move to ray data call
    mistral_tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1"
    )
    buffer_size = 40
    for classifier, classifier_spec in classifiers.items():
        classifier_spec["max_output_tokens"] = (
            max(
                len(mistral_tokenizer.encode(class_))
                for class_ in classifier_spec["classes"]
            )
            + buffer_size
        )

    # generate classifier prompts and responses
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
                concurrency=1,
                num_cpus=1,
            )
            .materialize()
        )

        # compute input/output distribution for Mistral model
        max_input_tokens = (
            ds.select_columns([f"{classifier}_prompt_tokens"])
            .map(compute_num_tokens, fn_kwargs={"col": f"{classifier}_prompt_tokens"})
            .max(on="num_tokens")
        )
        max_output_tokens = classifier_spec["max_output_tokens"]
        print(f"{classifier=} {max_input_tokens=} {max_output_tokens=}")
        max_model_length = max_input_tokens + max_output_tokens
        classifier_spec["max_model_length"] = max_model_length

    # generate classifier responses
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
                batch_size=80,
                num_gpus=num_mistral_workers_per_classifier,
                concurrency=1,
                accelerator_type=NVIDIA_TESLA_A10G,
            )
            .map(
                MistralDeTokenizer,
                fn_kwargs={"key": f"{classifier}_response"},
                concurrency=1,
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

    # write out to db
    mongo_bulk_op: MongoBulkInsert | MongoBulkUpdate
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
            batch_size=80,
            concurrency=10,
            num_cpus=0.1,
            zero_copy_batch=True,
        )
        .materialize()
    )
