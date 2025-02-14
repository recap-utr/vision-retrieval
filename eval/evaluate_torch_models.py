from model import ImageEmbeddingGraph
from new_evaluation import Evaluation
import json
from glob import glob
from typing import Callable
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from tqdm import tqdm
from time import time

OUTPUT_BASEPATH = "../data/eval_all/results/torch_models"


def standard_file_name(path: str) -> str:
    return "/".join(path.split("/")[-2:]).split(".")[0].lower()


def extract_query_name(path: str) -> str:
    return path.split("/")[-1].split(".")[0].lower()


def build_ground_truth(query_path: str) -> dict:
    with open(query_path, "r") as f:
        data = json.load(f)
    return data["userdata"]["cbrEvaluations"][0]["ranking"]


def build_mac_results(eval_json_path: str) -> dict[str, list[str]]:
    with open(eval_json_path, "r") as f:
        data = json.load(f)
    out = {}
    for query_name in data["individual"].keys():
        out[query_name] = [
            case["id"] for case in data["individual"][query_name]["mac"]["results"]
        ]
    return out


def embedding_func(model_path: str, base_model: str):
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


def build_ideal_mac_results(eval_json_path: str) -> dict[str, list[str]]:
    with open(eval_json_path, "r") as f:
        return json.load(f)


def run_eval(
    queries_path_glob: str,
    casebase_path_glob: str,
    eval_json_path: str,
    embedding_func: Callable[..., torch.Tensor],
    query_mapping: Callable[[str], str],
    casebase_mapping: Callable[[str], str],
) -> dict:
    query_files = glob(queries_path_glob)
    queries = {
        extract_query_name(q): ImageEmbeddingGraph(
            q, query_mapping(q), embedding_func, name=extract_query_name(q)
        )
        for q in tqdm(query_files)
    }
    print(f"Processed {len(queries)} queries")
    ground_truths = {k: build_ground_truth(q.graph_path) for k, q in queries.items()}
    # mac_results = build_mac_results(eval_json_path) # for real mac results
    mac_results = build_ideal_mac_results(eval_json_path)
    print("Processed ground truths and mac_results")
    casebase_files = glob(casebase_path_glob)
    print("Processing casebase...")
    start = time()
    casebase = {
        standard_file_name(q): ImageEmbeddingGraph(
            q, casebase_mapping(q), embedding_func, name=standard_file_name(q)
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


if __name__ == "__main__":
    MODEL_BASEPATH = "../vis_models/"
    model_names = [
        "logical_ft_arg",
        "logical_pt",
        "srip_ft_arg",
        "srip_pt",
        "treemaps_ft_arg",
        "treemaps_pt",
    ]
    BASE_MODEL = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft"

    res = {}
    RUNS = 10
    for run in range(RUNS):
        print(f"Starting run {run}/{RUNS}...")
        for t in ["simple", "complex"]:
            for model in model_names:
                print(f"Running {model} on {t}...")
                embedd = embedding_func(MODEL_BASEPATH + model, BASE_MODEL)
                model_type = model.split("_")[0]
                results = run_eval(
                    f"../data/retrieval_queries/microtexts-retrieval-{t}/*.json",
                    "../data/graphs/microtexts/*.json",
                    f"../data/eval_all/ideal_mac_{t}.json",
                    embedd,
                    lambda x: f"../data/eval_all/microtexts-retrieval-{t}/{model_type}/"
                    + x.split("/")[-1].split(".")[0]
                    + ".png",
                    lambda x: f"../data/eval_all/casebase/{model_type}/"
                    + x.split("/")[-1].split(".")[0]
                    + ".png",
                )
                res[model] = results
            with open(
                f"{OUTPUT_BASEPATH}/results_{t}_{run}.json",
                "w",
            ) as f:
                json.dump(res, f)
