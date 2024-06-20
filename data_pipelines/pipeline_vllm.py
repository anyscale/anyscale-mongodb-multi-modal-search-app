import numpy as np
import ray
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.multimodal.image import ImageFeatureData
from openai import OpenAI
from transformers import AutoTokenizer

HF_MODEL = "meta-llama/Llama-2-7b-chat-hf"

# Create a sampling params object.
sampling_params = SamplingParams(
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
    # top_k=None,
    seed=None,
    # stop=["<|eot_id|>", "<|end_of_text|>"],
)

num_llm_instances = 1

num_gpus_per_instance = 1

model_name_to_args = {
    "meta-llama/Llama-2-70b-chat-hf": {"max_model_len": 4096},
    "mistralai/Mistral-7B-Instruct-v0.1": {"max_model_len": 16832},
    "google/gemma-7b-it": {"max_model_len": 2432},
    "mlabonne/NeuralHermes-2.5-Mistral-7B": {"max_model_len": 16800},
}

# Mapping of model name to input prompt format.
# model_name_to_input_prompt_format = {
#     "meta-llama/Llama-2-7b-chat-hf": "[INST] {} [/INST]",
#     "mistralai/Mistral-7B-Instruct-v0.1": "[INST] {} [/INST]",
#     "google/gemma-7b-it": "<start_of_turn>model\n{}<end_of_turn>\n",
#     "mlabonne/NeuralHermes-2.5-Mistral-7B": "<|im_start|>system\nYou are a helpful assistant that will complete the sentence in the given input prompt.<|im_end|>\n<|im_start|>user{}<|im_end|>\n<|im_start|>assistant",
#     "meta-llama/Meta-Llama-3-8B-Instruct": (
#         "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. Complete the given prompt in several concise sentences.<|eot_id|>\n"
#         "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\n"
#         "<|start_header_id|>assistant<|end_header_id|>\n\n"
#     ),
# }


def construct_input_prompt(row, text_column):
    """Given the input row with raw text in `text_column` column,
    construct the input prompt for the model."""
    # prompt_format = model_name_to_input_prompt_format.get(HF_MODEL)
    # if prompt_format:
    #     row[text_column] = prompt_format.format(row[text_column])
    # return row
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    row[text_column] = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": row[text_column]}],
        add_generation_prompt=True,
        tokenize=False,
        return_tensors="np",
    )
    return row


# Create a class to do batch inference.
class LLMPredictor:
    def __init__(self, text_column):
        # Name of column containing the input text.
        self.text_column = text_column

        # Create an LLM.
        self.llm = LLM(
            model=HF_MODEL,
            **model_name_to_args.get(HF_MODEL, {}),
        )

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch[self.text_column], sampling_params)
        prompt = []
        generated_text = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(" ".join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }


class LMMPredictor:
    def __init__(self, text_column, image_column):
        self.text_column = text_column
        self.image_column = image_column
        self.llm = LLM(
            model="llava-hf/llava-v1.6-mistral-7b-hf",
            image_input_type="pixel_values",
            image_token_id=32000,
            image_input_shape="1, 5, 3, 336, 336",
            image_feature_size=2928,
            image_size="4096, 4096",
            enable_flattening=True,
            min_decodes_per_prefil=24,
            disable_image_processor=False,
        )

    def __call__(self, batch):
        image = batch[self.image_column]
        outputs = self.llm.generate(
            {
                "prompt": "What is strange about this image?",
                "multi_modal_data": ImageFeatureData(image),
            }
        )
        prompt = []
        generated_text = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(" ".join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }

def query_endpoints(base_url: str, api_key: str, text: str, retries: int = 6) -> str:
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    while True:
        try:
            response = client.chat.completions.create(
                model=HF_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                temperature=0,
                stream=False,
                max_tokens=2048,
            )
        except Exception as e:
            retries -= 1
            if retries == 0:
                raise e
            continue
        break

    return str(response.choices[0].message.content)


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


if __name__ == "__main__":
    text = "What is strange about this image?"
    image_url = "https://llava.hliu.cc/file=/nobackup/haotian/tmp/gradio/ca10383cc943e99941ecffdc4d34c51afb2da472/extreme_ironing.jpg"
    expected_output = query_llava(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key="esecret_wvfv1x446u8ifxujuqkimm7wjw",
        text=text,
        image_url=image_url,
    )
    print(f"{expected_output=}")

    df = pd.DataFrame({"text": [text], "image": [image_url]})
    ds = ray.data.from_pandas(df)
    ds = ds.map(
        construct_input_prompt,
        fn_kwargs={"text_column": "text"},
    )
    ds = ds.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_llm_instances,
        # Specify the number of GPUs required per LLM instance.
        num_gpus=num_gpus_per_instance,
        # Specify the batch size for inference. Set the batch size to as large
        # as possible without running out of memory.
        # If you encounter CUDA out-of-memory errors, decreasing
        # batch_size may help.
        batch_size=1,
        # Pass keyword arguments for the LLMPredictor class.
        fn_constructor_kwargs={"text_column": "text"},
        # Select the accelerator type.
        accelerator_type="A10G",
    )

    # Write inference output data out as Parquet files to S3.
    # Multiple files would be written to the output destination,
    # and each task would write one or more files separately.
    out = ds.to_pandas()
    print(f"vllm output {out}")


    # text = "What is the capital of france?"
    # expected_output = query_endpoints(
    #     base_url="https://api.endpoints.anyscale.com/v1",
    #     api_key="esecret_wvfv1x446u8ifxujuqkimm7wjw",
    #     text=text,
    # )
    # print(f"{expected_output=}")

    # # Apply batch inference for all input data.
    # df = pd.DataFrame({"text": [text]})
    # ds = ray.data.from_pandas(df)
    # ds = ds.map(
    #     construct_input_prompt,
    #     fn_kwargs={"text_column": "text"},
    # )
    # ds = ds.map_batches(
    #     LLMPredictor,
    #     # Set the concurrency to the number of LLM instances.
    #     concurrency=num_llm_instances,
    #     # Specify the number of GPUs required per LLM instance.
    #     num_gpus=num_gpus_per_instance,
    #     # Specify the batch size for inference. Set the batch size to as large
    #     # as possible without running out of memory.
    #     # If you encounter CUDA out-of-memory errors, decreasing
    #     # batch_size may help.
    #     batch_size=1,
    #     # Pass keyword arguments for the LLMPredictor class.
    #     fn_constructor_kwargs={"text_column": "text"},
    #     # Select the accelerator type.
    #     accelerator_type="A10G",
    # )

    # # Write inference output data out as Parquet files to S3.
    # # Multiple files would be written to the output destination,
    # # and each task would write one or more files separately.
    # out = ds.to_pandas()
    # print(f"vllm output {out}")
