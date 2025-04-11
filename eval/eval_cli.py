import argparse
import json
from glob import glob
from PIL import Image
import base64
import io
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.completion_usage import CompletionUsage
from correctness_completeness import _correctness_completeness_single
from ranx import Run, Qrels, evaluate
import statistics
from tqdm import tqdm
import copy
import os
from time import time
import typer
from typing_extensions import Annotated
from enum import Enum
from pathlib import Path
from model import ImageEmbeddingGraph
from new_evaluation import Evaluation
from glob import glob
from typing import Callable, List, cast
import torch
from transformers import AutoModel, AutoImageProcessor

app = typer.Typer()


class Mode(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


load_dotenv()  # take environment variables from .env.

client = OpenAI()
IMG_PLACEHOLDER = "data:image/png;base64,"


@app.command()
def eval_oai(
    openai_model: Annotated[
        str,
        typer.Argument(help="The fine-tuned OpenAI model to be evaluated"),
    ],
    mode: Mode,
    queries_path: Annotated[
        Path,
        typer.Argument(
            help="This path should contain the queries as arguebuf-compatible files (e.g. JSON) as well as the corresponding images with the same name (e.g. graph1.png for graph1.json)."
        ),
    ],
    casebase_images_path: Annotated[
        Path,
        typer.Argument(
            help="This path should contain the images for the casebase argument graphs with the same name (e.g. case1.png for case1.json)."
        ),
    ],
    mac_results_path: Path,
    output_path: Path,
):
    ranking_path = output_path / "ranking.json"
    results_path = output_path / "results.json"

    res = {}
    ranking_oai = {}
    qrels = {}
    run = {}
    sum_tokens = 0
    query_names = [
        path.split("/")[-1].split(".")[0] for path in glob(f"{queries_path}/*.json")
    ]
    loaded_existing = os.path.exists(ranking_path)
    # mac_results = build_mac_results(MAC_RESULTS_PATH)
    mac_results = build_ideal_mac_results(mac_results_path)
    print(f"Using these MAC results: {mac_results_path}")
    request_durations = []
    images_sent = []
    if loaded_existing:
        with open(ranking_path, "r") as f:
            ranking_oai = json.load(f)
            print("Loaded existing ranking")
            sum_tokens = ranking_oai["sum_tokens"]
            request_durations = ranking_oai["request_durations"]
    else:
        print(
            f"No existing ranking found. Starting new evaluation with {len(query_names)} {mode} queries..."
        )
    for query_name in tqdm(query_names):
        query_text = f"{queries_path}/{query_name}.json"
        query_image = f"{queries_path}/{query_name}.png"

        with open(query_text, "r") as f:
            query_text = json.load(f)
        reference_rankings = {
            k.split("/")[1]: 4 - v
            for k, v in query_text["userdata"]["cbrEvaluations"][0]["ranking"].items()
        }
        qrels[query_name] = reference_rankings
        casebase_images = [
            f"{casebase_images_path}/{n}.png" for n in mac_results[query_name]
        ]
        images_sent.append(casebase_images)
        if loaded_existing and query_name in ranking_oai:
            answer = ranking_oai[query_name]
            print(f"{query_name}: Used existing ranking")
        else:
            prompt_messages = [
                {
                    "role": "system",
                    "content": "You are an assistant that specializes in retrieving argument graphs based on their visualizations.",
                },
                {
                    "role": "user",
                    "content": f"Take a look at the following images in space reclaiming icicle chart visualization. Image 1 represents the query, images 2-{1 + len(reference_rankings.keys())} are retrieval candidates. Please rank all retrieval candidates (images 2-{1 + len(reference_rankings.keys())}) in descending order based on their similarity to the query image. If there were, for example, the three retrieval candidates images 2-4, image 3 having the highest similarity, image 2 the second highest and image 4 the lowest, you would output: 3,2,4.",
                },
                {"role": "user", "content": "This is image 1, the query"},
            ]
            prompt_messages.append(append_images(query_image))
            prompt_messages.append(
                {
                    "role": "user",
                    "content": f"The following are images 2-{1 + len(casebase_images)}, retrieval candidates:",
                }
            )
            for img in casebase_images:
                prompt_messages.append(append_images(img))
            prompt_messages.append(
                {
                    "role": "user",
                    "content": "The correct order of the retrieval images is:",
                }
            )
            request_start = time()
            completion = client.chat.completions.create(
                model=openai_model,
                messages=prompt_messages,
                temperature=0.0,
            )
            request_durations.append(time() - request_start)
            answer = completion.choices[0].message.content
            usage = completion.usage
            usage = cast(CompletionUsage, usage)
            sum_tokens += usage.total_tokens
        try:
            keys = [key.split("/")[-1] for key in mac_results[query_name]]
            answer = cast(str, answer)
            order = [int(i) for i in answer.split(",")]
            # assert len(set(order)) == len(
            #     reference_rankings
            # ), "Rankings must be unique and equal to the number of images"
            # assert set(order) == set(
            #     range(2, len(reference_rankings) + 2)
            # ), "Rankings must be a permutation of 1 to N"
        except ValueError:
            order = []
        simulated_sims = {
            keys[idx - 2]: 1 - 0.1 * (pos + 1) for pos, idx in enumerate(order)
        }
        run[query_name] = simulated_sims
        # stattdessen Evaluation(query)
        ranking_oai[query_name] = answer
        res[query_name] = {
            "correctness_completeness": _correctness_completeness_single(
                reference_rankings, simulated_sims, None
            )
        }
    print(res)
    ranking_oai["sum_tokens"] = sum_tokens
    ranking_oai["request_durations"] = request_durations
    ranking_oai["images_sent"] = images_sent
    with open(ranking_path, "w") as f:
        json.dump(ranking_oai, f)
    print("Starting evaluation")
    start = time()
    run = Run(run)
    evaluate(
        Qrels(qrels),
        run,
        ["ndcg_burges", "ndcg", "map", "f1", "recall", "precision"],
        return_mean=False,
    )
    results = results_add_correctness_completeness(
        query_names, copy.deepcopy(run.mean_scores), qrels, run, None
    )
    results["sum_tokens"] = sum_tokens
    results["request_duration"] = sum(request_durations)
    with open(results_path, "w") as f:
        json.dump(results, f)
    print(results)
    print(f"Evaluation duration: {time() - start}")


def results_add_correctness_completeness(
    queries: list[str],
    results: dict[str, float],
    qrels: dict[str, dict[str, int]],
    run: Run,
    k: int | None,
):
    correctness, completeness = [], []
    for queryname in queries:
        corr, comp = _correctness_completeness_single(
            qrels[queryname], run[queryname], k
        )
        correctness.append(corr)
        completeness.append(comp)
    results["correctness"] = statistics.mean(correctness)
    results["completeness"] = statistics.mean(completeness)
    return results


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return IMG_PLACEHOLDER + base64.b64encode(buffered.getvalue()).decode("utf-8")


def build_mac_results(eval_json_path: str) -> dict[str, list[str]]:
    with open(eval_json_path, "r") as f:
        data = json.load(f)
    out = {}
    for query_name in data["individual"].keys():
        out[query_name] = [
            case["id"] for case in data["individual"][query_name]["mac"]["results"]
        ]
    return out


def build_ideal_mac_results(eval_json_path: Path) -> dict[str, list[str]]:
    with open(eval_json_path, "r") as f:
        res = {}
        data = json.load(f)
        for k, v in data.items():
            res[k] = [name.split("/")[-1] for name in v]
        return res


def append_images(image_path: str) -> dict:
    # mac phase has this prefix, eval folder structure does not
    image_path = image_path.replace("microtexts/", "")
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_base64(Image.open(image_path).convert("RGB"))
                },
            }
        ],
    }


def standard_file_name(path: str) -> str:
    return "/".join(path.split("/")[-2:]).split(".")[0].lower()


def extract_query_name(path: str) -> str:
    return path.split("/")[-1].split(".")[0].lower()


def build_ground_truth(query_path: str | Path) -> dict:
    with open(query_path, "r") as f:
        data = json.load(f)
    return {
        k.split("/")[-1]: v
        for k, v in data["userdata"]["cbrEvaluations"][0]["ranking"].items()
    }


def embedding_func(model_path: Path, base_model: str):
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(base_model)

    def func(image: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            inputs = processor(image, return_tensors="pt")
            outputs = model(**inputs)
            outputs = outputs.pooler_output
            return outputs

    return func


def get_image_path(images_path: Path, q: str) -> Path:
    return images_path / (q.split(".")[0] + ".png")


def _run_torch_eval(
    queries_path: Path,
    casebase_argument_graphs_path: Path,
    mac_results_path: Path,
    embedding_func: Callable[..., torch.Tensor],
) -> dict:
    query_files = [f for f in os.listdir(queries_path) if f.endswith(".json")]
    queries = {
        extract_query_name(q): ImageEmbeddingGraph(
            queries_path / q,
            get_image_path(queries_path, q),
            embedding_func,
            name=extract_query_name(q),
        )
        for q in tqdm(query_files)
    }
    print(f"Processed {len(queries)} queries")
    ground_truths = {k: build_ground_truth(q.graph_path) for k, q in queries.items()}
    # mac_results = build_mac_results(eval_json_path) # for real mac results
    mac_results = build_ideal_mac_results(mac_results_path)
    print("Processed ground truths and mac_results")
    casebase_files = [
        f for f in os.listdir(casebase_argument_graphs_path) if f.endswith(".json")
    ]
    print("Processing casebase...")
    start = time()
    casebase = {
        standard_file_name(q): ImageEmbeddingGraph(
            casebase_argument_graphs_path / q,
            get_image_path(casebase_argument_graphs_path, q),
            embedding_func,
            name=standard_file_name(q),
        )
        for q in tqdm(casebase_files)
    }
    duration = time() - start
    print(f"Processed {len(casebase)} casebase files in {duration} seconds")
    print("Starting eval...")
    ev = Evaluation(
        casebase, ground_truths, mac_results, list(queries.values()), times=False
    )
    res = ev.as_dict()
    res["embedding_time"] = duration
    return res


@app.command()
def eval_torch(
    models: List[Path],
    queries_path: Annotated[
        Path,
        typer.Argument(
            help="The folder which contains the query argument graphs. Expected to contain a 'simple' and a 'complex' subdirectory which contain the queries of the respective data sets in arguebuf-compatible JSON files and the corresponding images with the same name."
        ),
    ],
    casebase_arguments_path: Annotated[
        Path,
        typer.Argument(
            help="This directory should include all casebase argument graphs in an arguebuf-compatible JSON format, e.g. JSON and the corresponding images with the same name"
        ),
    ],
    mac_results_path: Path,
    output_path: Annotated[Path, typer.Argument(help="Filename of the resulting JSON")],
    base_model: str = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
    num_runs: int = 1,
):
    res = {}
    for run in range(num_runs):
        print(f"Starting run {run}/{num_runs}...")
        for model in models:
            print(f"Evaluating {model}...")
            embedd = embedding_func(model, base_model)
            results = _run_torch_eval(
                queries_path,
                casebase_arguments_path,
                mac_results_path,
                embedd,
            )
            res[str(model.resolve())] = results
            with open(
                output_path,
                "w",
            ) as f:
                json.dump(res, f)


if __name__ == "__main__":
    app()
