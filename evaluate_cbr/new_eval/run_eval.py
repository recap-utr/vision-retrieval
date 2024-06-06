from model import ImageGraph, ImageEmbeddingGraph
from new_evaluation import Evaluation
import json
from glob import glob
from typing import Callable
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from tqdm import tqdm
import sys


def standard_file_name(path: str) -> str:
    return "".join(path.split("/")[-2:]).split(".")[0].lower()


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
    processor = AutoImageProcessor.from_pretrained(base_model)

    def func(image: Image.Image) -> torch.Tensor:
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        return outputs.pooler_output

    return func


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
        standard_file_name(q): ImageGraph(q, query_mapping(q)) for q in query_files
    }
    queries = {k: ImageEmbeddingGraph(v, embedding_func) for k, v in queries.items()}
    print("Processed queries")
    ground_truths = {k: build_ground_truth(q.graph_path) for k, q in queries.items()}
    mac_results = build_mac_results(eval_json_path)
    print("Processed ground truths and mac_results")
    casebase_files = glob(casebase_path_glob)[:10]
    casebase = {
        standard_file_name(q): ImageEmbeddingGraph(
            ImageGraph(q, casebase_mapping(q)), embedding_func
        )
        for q in casebase_files
    }
    print("Processed casebase. Starting eval...")
    ev = Evaluation(casebase, ground_truths, mac_results, list(queries.values()))
    return ev.as_dict()


if __name__ == "__main__":

    MODEL_PATH = "/home/kilian/vision-retrieval/models/srip_best"
    BASE_MODEL = "microsoft/swinv2-tiny-patch4-window8-256"

    embedd = embedding_func(MODEL_PATH, BASE_MODEL)
    run_eval(
        "/home/kilian/vision-retrieval/data/retrieval_query/microtexts-retrieval-complex/*.json",
        "/home/kilian/vision-retrieval/data/graphs/microtexts/*.json",
        "/home/kilian/vision-retrieval/evaluate_cbr/eval.json",
        embedd,
        lambda x: "/home/kilian/vision-retrieval/data/eval_all/microtexts-retrieval-complex/srip/"
        + x.split("/")[-1].split(".")[0]
        + ".png",
        lambda x: "/home/kilian/vision-retrieval/data/eval_all/casebase/srip/"
        + x.split("/")[-1].split(".")[0]
        + ".png",
    )
