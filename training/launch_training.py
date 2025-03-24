import typer

from pretraining import main as pt
from finetuning import main as ft
from typing_extensions import Annotated
from pathlib import Path

app = typer.Typer()


@app.command()
def pretraining(
    dataset: Annotated[
        str,
        typer.Argument(
            help="A HuggingFace dataset. Can be a local path or a dataset hub path"
        ),
    ],
    max_epochs: Annotated[
        int,
        typer.Argument(
            help="Maximum number of epochs to train for. The model will be trained for at most this many epochs, but may stop early."
        ),
    ],
    save_path: Path,
    base_model: Annotated[
        str,
        typer.Option(
            help="The base model to use for pretraining'"
        ),
    ] = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
    latent_dim: Annotated[
        int,
        typer.Option(
            help="Dimensionality of latent representation of the base model, e.g. 1536"
        ),
    ] = 1536,
    batch_size: int = 32,
    
    wandb_project_name: Annotated[
        str, typer.Option(help="Leave blank if no wandb logging is needed")
    ] = "",
    num_workers: int = 30,
):
    pt(
        dataset,
        base_model,
        latent_dim,
        batch_size,
        max_epochs,
        save_path,
        wandb_project_name,
        num_workers,
    )


@app.command()
def finetuning(
    dataset: Annotated[
        str,
        typer.Argument(
            help="A HuggingFace dataset. Can be a local path or a dataset hub path"
        ),
    ],
    save_path: Path,
    pretrained_checkpoint: Annotated[
        str,
        typer.Argument(
            help="The pretrained model to use for finetuning. Can be a local path or a model hub path"
        ),
    ],
    batch_size: int,
    max_epochs: Annotated[
        int,
        typer.Argument(
            help="Maximum number of epochs to train for. The model will be trained for at most this many epochs, but may stop early."
        ),
    ],
    base_model: Annotated[
        str,
        typer.Argument(
            help="The base model to use for pretraining, e.g. microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft'"
        ),
    ] = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
    latent_dim: Annotated[
        int,
        typer.Argument(
            help="Dimensionality of latent representation of the base model, e.g. 1536"
        ),
    ] = 1536,
    wandb_project_name: Annotated[
        str, typer.Option(help="Leave blank if no wandb logging is needed")
    ] = "",
    num_workers: int = 30,
):
    ft(
        base_model,
        batch_size,
        latent_dim,
        max_epochs,
        save_path,
        pretrained_checkpoint,
        dataset,
        wandb_project_name,
        num_workers,
    )


if __name__ == "__main__":
    app()
