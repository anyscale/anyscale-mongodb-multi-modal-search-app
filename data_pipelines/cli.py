"""CLI for running the data pipelines"""

from typing import Literal

import typer

from pydantic import BaseModel
from data_pipelines.data import clear_data_in_collection, setup_collection
from data_pipelines.offline_compute import run_pipeline as offline_run_pipeline
from data_pipelines.online_compute import run_pipeline as online_run_pipeline
from ray.util.accelerators import NVIDIA_TESLA_A10G

app = typer.Typer()


class ScalingConfig(BaseModel):
    num_llava_tokenizer_workers: int
    num_llava_model_workers: int
    llava_model_accelerator_type: str
    llava_model_batch_size: int

    num_mistral_tokenizer_workers_per_classifier: int
    num_mistral_model_workers_per_classifier: int
    num_mistral_detokenizer_workers_per_classifier: int
    mistral_model_batch_size: int
    mistral_model_accelerator_type: str

    num_embedder_workers: int
    embedding_model_batch_size: int
    embedding_model_accelerator_type: str

    db_update_batch_size: int
    num_db_workers: int

    @classmethod
    def estimate(nsamples: int) -> "ScalingConfig":
        if nsamples < 1_000:
            return ScalingConfig(
                num_llava_tokenizer_workers=1,
                num_llava_model_workers=1,
                llava_model_accelerator_type=NVIDIA_TESLA_A10G,
                llava_model_batch_size=80,
                num_mistral_tokenizer_workers_per_classifier=1,
                num_mistral_model_workers_per_classifier=1,
                num_mistral_detokenizer_workers_per_classifier=1,
                mistral_model_batch_size=80,
                mistral_model_accelerator_type=NVIDIA_TESLA_A10G,
                num_embedder_workers=1,
                embedding_model_batch_size=80,
                embedding_model_accelerator_type=NVIDIA_TESLA_A10G,
                db_update_batch_size=80,
                num_db_workers=1,
            )
        elif nsamples < 100_000:
            return ScalingConfig(
                num_llava_tokenizer_workers=2,
                num_llava_model_workers=5 * nsamples // 4_000,
                llava_model_accelerator_type=NVIDIA_TESLA_A10G,
                llava_model_batch_size=80,
                num_mistral_tokenizer_workers_per_classifier=2,
                num_mistral_model_workers_per_classifier=1 * nsamples // 4_000,
                num_mistral_detokenizer_workers_per_classifier=2,
                mistral_model_batch_size=80,
                mistral_model_accelerator_type=NVIDIA_TESLA_A10G,
                num_embedder_workers=1 * nsamples // 4_000,
                embedding_model_batch_size=80,
                embedding_model_accelerator_type=NVIDIA_TESLA_A10G,
                db_update_batch_size=80,
                num_db_workers=min(1 * nsamples // 4_000, 10),
            )
        else:
            raise NotImplementedError("More than 100k samples not supported yet")


@app.command()
def main(
    data_path: str = typer.Option(..., help="Path to the data file"),
    nsamples: int = typer.Option(1000, help="Number of samples to process"),
    db_name: str = "myntra",
    collection_name: str = "myntra-items-offline",
    inference_type: Literal["offline", "online"] = "offline",
    mode: Literal["first_run", "update"] = "update",
    cluster_size: str = "m0",
    scaling_config_path: str | None = None,
):
    if mode == "first_run":
        print("Setting up collection")
        setup_collection(collection_name, cluster_size)
        clear_data_in_collection(collection_name)

    if inference_type == "offline":
        
        if scaling_config_path is not None:
            with open(scaling_config_path, "r") as f:
                scaling_config = ScalingConfig.model_validate_json(f.read())
        else:
            scaling_config = ScalingConfig.estimate(nsamples)

        offline_run_pipeline(
            path=data_path,
            nsamples=nsamples,
            db_name=db_name,
            collection_name=collection_name,
            **scaling_config.model_dump(),
        )

    elif inference_type == "online":
        if nsamples > 1000:
            typer.echo(
                "Using online mode for more than 1000 samples will be slow and costly"
            )

            confirm = typer.confirm(
                "Are you sure you still want to run the online pipeline?",
                abort=True,
            )

            if not confirm:
                return

        online_run_pipeline(
            path=data_path,
            nsamples=nsamples,
            mode=mode,
            db_name=db_name,
            collection_name=collection_name,
        )


if __name__ == "__main__":
    app()
