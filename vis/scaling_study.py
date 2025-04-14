from glob import glob
from tqdm import tqdm
import arguebuf as ab
from pathlib import Path
from util import find_heighest_root_node
from time import time
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
import torch
import json
import typer
from typing import Annotated
from render import render, RenderMethod

"""
This script is used to generate the data for our scaling study.
"""
app = typer.Typer()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def embedding_func(model_path: str | Path, base_model: str):
    model = AutoModel.from_pretrained(model_path).to(DEVICE)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(base_model)

    def func(image: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            inputs = processor(image, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            outputs = outputs.pooler_output
            return outputs

    return func


@app.command()
def scaling_study(
    model_path: Path,
    visualization_method: RenderMethod,
    requests_graphs_glob: Annotated[
        str, typer.Argument(help="Glob pattern to find the request graphs")
    ],
    casebase: Annotated[
        Path, typer.Argument(help="Path to the single case base graph")
    ],
    queries_dest: Annotated[
        Path, typer.Argument(help="Path where the query images should be stored")
    ],
    case_dest: Annotated[
        Path, typer.Argument(help="Path where the case image should be stored")
    ],
    results_path: Annotated[
        Path, typer.Argument(help="JSON results path")
    ],
    execute_num_times: Annotated[
        int, typer.Argument(help="How many times should the experiment be repeated?")
    ] = 10,
    base_model: str = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
):

    visualization_times = {
        k.split("/")[-1].split(".")[0]: 0.0 for k in glob(requests_graphs_glob)
    }
    embedding_times = {
        k.split("/")[-1].split(".")[0]: 0.0 for k in glob(requests_graphs_glob)
    }
    similarity_computation_times = {
        k.split("/")[-1].split(".")[0]: 0.0 for k in glob(requests_graphs_glob)
    }
    similarity_computation_times2 = {
        k.split("/")[-1].split(".")[0]: 0.0 for k in glob(requests_graphs_glob)
    }
    embeddings = {}

    emb = embedding_func(model_path, base_model)

    print("measuring visualization times")
    for _ in tqdm(range(execute_num_times)):
        for file in glob(requests_graphs_glob):
            graph = ab.load.file(file)
            number_s_nodes = file.split("/")[-1].split(".")[0]
            path = Path(f"{queries_dest}/{file.split('/')[-1].replace('json', 'png')}")
            start = time()
            render(graph, path, normalize_height=True, method=visualization_method)
            visualization_times[number_s_nodes] += time() - start

    # visualize case
    case_graph = ab.load.file(casebase)
    case_path = Path(case_dest)
    render(case_graph, case_path, normalize_height=True, method=visualization_method)

    print("measuring embedding times")
    for _ in tqdm(range(execute_num_times)):

        for file in glob(f"{queries_dest}/*.png"):
            image = Image.open(file).convert("RGB")
            number_s_nodes = file.split("/")[-1].split(".")[0]
            start = time()
            embeddings[number_s_nodes] = emb(image)
            embedding_times[number_s_nodes] += time() - start

    # embedd case
    case_image = Image.open(case_dest).convert("RGB")
    case_emb = emb(case_image)

    print("measuring similarity computation times: cosine")
    for _ in tqdm(range(execute_num_times)):

        for number_s_nodes, emb in embeddings.items():
            start = time()
            torch.nn.functional.cosine_similarity(emb, case_emb)
            similarity_computation_times[number_s_nodes] += time() - start

    visualization_times = {
        k: v / execute_num_times for k, v in visualization_times.items()
    }
    embedding_times = {k: v / execute_num_times for k, v in embedding_times.items()}
    similarity_computation_times = {
        k: v / execute_num_times for k, v in similarity_computation_times.items()
    }

    print("measuring similarity computation times: dot")
    for i in tqdm(range(execute_num_times)):

        for number_s_nodes, emb in embeddings.items():
            start = time()
            torch.dot(emb.flatten(), case_emb.flatten())
            similarity_computation_times2[number_s_nodes] += time() - start

    visualization_times = {
        k: v / execute_num_times for k, v in visualization_times.items()
    }
    embedding_times = {k: v / execute_num_times for k, v in embedding_times.items()}
    similarity_computation_times2 = {
        k: v / execute_num_times for k, v in similarity_computation_times.items()
    }

    print("visualization times")
    print(visualization_times)
    print("embedding times")
    print(embedding_times)
    print("sim times: cosine")
    print(similarity_computation_times)
    print("sim times: dot")
    print(similarity_computation_times2)
    with open(results_path, "w") as f:
        json.dump(
            {
                "visualization_times": visualization_times,
                "embedding_times": embedding_times,
                "similarity_computation_times": similarity_computation_times,
                "similarity_computation_times2": similarity_computation_times2,
            },
            f,
        )
    print("successfully saved results. Done!")


if __name__ == "__main__":
    app()
