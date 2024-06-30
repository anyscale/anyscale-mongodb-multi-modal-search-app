"""CLI for running the data pipelines"""

from typing import Literal

import typer

from data_pipelines.data import clear_data_in_collection, setup_collection
from data_pipelines.offline_compute import run_pipeline as offline_run_pipeline
from data_pipelines.online_compute import run_pipeline as online_run_pipeline

app = typer.Typer()


@app.command()
def main(
    data_path: str,
    nsamples: int,
    db_name: str = "myntra",
    collection_name: str = "myntra-items-offline",
    inference_type: Literal["offline", "online"] = "offline",
    mode: Literal["first_run", "update"] = "update",
    cluster_size: str = "m0",
):
    if mode == "first_run":
        print("Setting up collection")
        setup_collection(collection_name, cluster_size)
        clear_data_in_collection(collection_name)

    if inference_type == "offline":
        offline_run_pipeline(
            path=data_path,
            nsamples=nsamples,
            db_name=db_name,
            collection_name=collection_name,
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
