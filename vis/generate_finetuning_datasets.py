from render import render_command, RenderMethod
from multiprocessing import Pool, TimeoutError
import os
import typer
from typing_extensions import Annotated
from pathlib import Path
from time import time
from tqdm import tqdm
import signal

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
        start = time()
        tasks = []
        for dataset_folder in input_folder.iterdir():
            if not dataset_folder.is_dir():
                continue
            dataset_name = dataset_folder.name
            print(f"Processing dataset {dataset_name}")
            dataset_output_folder = output_folder / dataset_name
            os.makedirs(dataset_output_folder, exist_ok=True)
            tasks.append((dataset_folder, dataset_output_folder, method))
        pool.starmap(_workload, tasks)
        end = time()
        print(f"Finished processing {len(tasks)} datasets in {end - start:.2f} seconds.")

@app.command()
def single_dataset(
    input_folder: Annotated[
        Path,
        typer.Argument(
            help="This folder should contain one direct subfolder per dataset. Each graph should only be stored a single time, e.g. in JSON format."
        ),
    ],
    output_folder: Path,
    timeout_folder: Path,
    method: RenderMethod = RenderMethod.SRIP2,
    num_processes: Annotated[
        int,
        typer.Argument(
            help="Number of threads to use. Only useful when visualizing a large number of datasets as one thread processes one dataset."
        ),
    ] = 1,
):
    datasets = [f for f in input_folder.iterdir() if f.is_dir()]
    files = [f for ds in datasets for f in ds.iterdir()]
    os.makedirs(output_folder, exist_ok=True)
    start = time()
    tasks = [(file, output_folder / (file.stem + ".png"), method) for file in files]

    try:
        with Pool(num_processes) as pool:
            results = [pool.apply_async(_workload_single_dataset, task) for task in tasks]

            # Process results with timeout in the main process
            for i, result in enumerate(tqdm(results, desc="Processing files")):
                try:
                    result.get(timeout=10)  
                except TimeoutError:
                    print(f"Task {i} timed out, moving to timeout folder.")
                    os.replace(tasks[i][0], timeout_folder / tasks[i][0].name)
                except Exception as e:
                    print(f"Error in task {i}: {e}")
    except Exception as e:
        print(f"Pool processing error: {e}")

    end = time()
    print(f"Finished processing {len(tasks)} datasets in {end - start:.2f} seconds.")

def _workload_single_dataset(input_file: Path, output_file: Path, method: RenderMethod):
    # Remove signal handling, rely on the parent process timeout
    print(f"Processing {input_file}")
    try:
        if os.path.exists(output_file):
            print(f"File {output_file} already exists, skipping.")
            return True

        render_command(input_file, output_file, method=method)
        return True
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False



def _workload(input_folder: Path, output_folder: Path, method: RenderMethod):
    for file in input_folder.iterdir():
        print(f"Processing {file}")
        if file.is_dir():
            continue
        try:
            render_command(file, output_folder / (file.stem + ".png"), method=method)
        except Exception as e:
            print(f"Error processing {file}: {e}")


if __name__ == "__main__":
    app()
