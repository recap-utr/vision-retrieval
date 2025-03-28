from render import render_command, RenderMethod
from multiprocessing import Pool
import os
import typer
from typing_extensions import Annotated
from pathlib import Path

app = typer.Typer()


@app.command()
def generate_datasets_command(
    input_folder: Annotated[
        Path,
        typer.Argument(
            help="This folder should contain one direct subfolder per dataset. Each graph should only be stored a single time, e.g. in JSON format."
        ),
    ],
    output_folder: Path,
    method: RenderMethod = RenderMethod.SRIP2,
    num_processes: Annotated[
        int,
        typer.Argument(
            help="Number of threads to use. Only useful when visualizing a large number of datasets as one thread processes one dataset."
        ),
    ] = 1,
):
    with Pool(num_processes) as pool:
        tasks = []
        for dataset_folder in input_folder.iterdir():
            if not dataset_folder.is_dir():
                continue
            dataset_name = dataset_folder.name
            print(f"Processing dataset {dataset_name}")
            output_folder = output_folder / dataset_name
            os.makedirs(output_folder, exist_ok=True)
            tasks.append((dataset_folder, output_folder, method))
        pool.starmap(_workload, tasks)


def _workload(input_folder: Path, output_folder: Path, method: RenderMethod):
    for file in input_folder.iterdir():
        print(f"Processing {file}")
        if file.is_dir():
            continue
        render_command(file, output_folder / (file.stem + ".png"), method=method)


if __name__ == "__main__":
    app()
