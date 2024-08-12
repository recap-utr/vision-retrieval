from model import ImageGraph, ImageEmbeddingGraph
from new_evaluation import Evaluation
import json
from glob import glob
from typing import Callable
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from tqdm import tqdm
import pandas as pd
from time import time

# TODO: fix queries (currently none found), run, eval results


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


topic_mapping = {
    "allow_shops_to_open_on_holidays_and_sundays": [
        "microtexts/nodeset6375",
        "microtexts/nodeset6410",
        "microtexts/nodeset6419",
        "microtexts/nodeset6449",
        "microtexts/nodeset6451",
        "microtexts/nodeset6457",
        "microtexts/nodeset6462",
        "microtexts/nodeset6466",
    ],
    "health_insurance_cover_complementary_medicine": [
        "microtexts/nodeset6363",
        "microtexts/nodeset6370",
        "microtexts/nodeset6373",
        "microtexts/nodeset6378",
        "microtexts/nodeset6385",
        "microtexts/nodeset6386",
        "microtexts/nodeset6395",
        "microtexts/nodeset6412",
        "med1",
        "med2",
        "med3",
        "med4",
    ],
    "higher_dog_poo_fines": [
        "microtexts/nodeset6362",
        "microtexts/nodeset6367",
        "microtexts/nodeset6371",
        "microtexts/nodeset6392",
        "microtexts/nodeset6400",
        "microtexts/nodeset6420",
        "microtexts/nodeset6452",
        "microtexts/nodeset6468",
        "dog1",
        "dog2",
        "dog3",
        "dog4",
    ],
    "introduce_capital_punishment": [
        "microtexts/nodeset6366",
        "microtexts/nodeset6383",
        "microtexts/nodeset6387",
        "microtexts/nodeset6391",
        "microtexts/nodeset6450",
        "microtexts/nodeset6453",
        "microtexts/nodeset6464",
        "microtexts/nodeset6469",
        "death1",
        "death2",
        "death3",
        "death4",
    ],
    "public_broadcasting_fees_on_demand": [
        "microtexts/nodeset6364",
        "microtexts/nodeset6374",
        "microtexts/nodeset6389",
        "microtexts/nodeset6446",
        "microtexts/nodeset6454",
        "microtexts/nodeset6463",
        "microtexts/nodeset6470",
        "media1",
        "media2",
        "media3",
        "media4",
    ],
    "cap_rent_increases": [
        "microtexts/nodeset6369",
        "microtexts/nodeset6377",
        "microtexts/nodeset6384",
        "microtexts/nodeset6418",
        "microtexts/nodeset6455",
        "microtexts/nodeset6465",
        "rent1",
        "rent2",
        "rent3",
        "rent4",
        "cap_rent_increases",
    ],
    "charge_tuition_fees": [
        "microtexts/nodeset6381",
        "microtexts/nodeset6388",
        "microtexts/nodeset6394",
        "microtexts/nodeset6407",
        "microtexts/nodeset6447",
        "microtexts/nodeset6456",
        "tuition1",
        "tuition2",
        "tuition3",
        "tuition4",
        "charge_tuition_fees",
    ],
    "keep_retirement_at_63": [
        "microtexts/nodeset6382",
        "microtexts/nodeset6409",
        "microtexts/nodeset6411",
        "microtexts/nodeset6416",
        "microtexts/nodeset6421",
        "microtexts/nodeset6461",
    ],
    "over_the_counter_morning_after_pill": [
        "microtexts/nodeset6368",
        "microtexts/nodeset6397",
        "microtexts/nodeset6402",
        "microtexts/nodeset6406",
        "microtexts/nodeset6414",
    ],
    "increase_weight_of_BA_thesis_in_final_grade": [
        "microtexts/nodeset6376",
        "microtexts/nodeset6408",
        "microtexts/nodeset6448",
        "microtexts/nodeset6467",
    ],
    "stricter_regulation_of_intelligence_services": [
        "microtexts/nodeset6365",
        "microtexts/nodeset6401",
        "microtexts/nodeset6405",
        "microtexts/nodeset6458",
    ],
    "EU_influence_on_political_events_in_Ukraine": [
        "microtexts/nodeset6399",
        "microtexts/nodeset6415",
        "microtexts/nodeset6460",
        "eu_influence_on_political_events_in_ukraine",
    ],
    "make_video_games_olympic": [
        "microtexts/nodeset6380",
        "microtexts/nodeset6396",
        "microtexts/nodeset6417",
    ],
    "school_uniforms": [
        "microtexts/nodeset6372",
        "microtexts/nodeset6390",
        "microtexts/nodeset6398",
    ],
    "TXL_airport_remain_operational_after_BER_opening": [
        "microtexts/nodeset6403",
        "microtexts/nodeset6422",
        "microtexts/nodeset6459",
    ],
    "buy_tax_evader_data_from_dubious_sources": [
        "microtexts/nodeset6379",
        "microtexts/nodeset6404",
    ],
    "partial_housing_development_at_Tempelhofer_Feld": [
        "microtexts/nodeset6393",
        "microtexts/nodeset6413",
    ],
    "waste_separation": ["microtexts/nodeset6361"],
    "other": [
        "microtexts/nodeset6423",
        "microtexts/nodeset6424",
        "microtexts/nodeset6425",
        "microtexts/nodeset6426",
        "microtexts/nodeset6427",
        "microtexts/nodeset6428",
        "microtexts/nodeset6429",
        "microtexts/nodeset6430",
        "microtexts/nodeset6431",
        "microtexts/nodeset6432",
        "microtexts/nodeset6433",
        "microtexts/nodeset6434",
        "microtexts/nodeset6435",
        "microtexts/nodeset6436",
        "microtexts/nodeset6437",
        "microtexts/nodeset6438",
        "microtexts/nodeset6439",
        "microtexts/nodeset6440",
        "microtexts/nodeset6441",
        "microtexts/nodeset6442",
        "microtexts/nodeset6443",
        "microtexts/nodeset6444",
        "microtexts/nodeset6445",
    ],
}


def simulate_mac_phase(queries: list[ImageGraph]) -> dict[str, list[str]]:
    res = {}
    for query in queries:
        q_name = query.name.split(".")[0]
        for k, v in topic_mapping.items():
            if q_name.lower() == k.lower():
                res[q_name.lower()] = [k for k in v]
                break
            if q_name in v:
                res[q_name.lower()] = [k for k in v]
                break
    return res


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
    mac_results = build_mac_results(eval_json_path)
    # mac_results = simulate_mac_phase(list(queries.values()))
    print("Processed ground truths and mac_results")
    casebase_files = glob(casebase_path_glob)
    start = time()
    casebase = {
        standard_file_name(q): ImageEmbeddingGraph(
            q, casebase_mapping(q), embedding_func, name=standard_file_name(q)
        )
        for q in tqdm(casebase_files)
    }
    print(f"Processed {len(casebase)} casebase files in {time()-start} seconds")
    print("Starting eval...")
    ev = Evaluation(
        casebase, ground_truths, mac_results, list(queries.values()), times=True
    )
    return ev.as_dict()


if __name__ == "__main__":

    MODEL_BASEPATH = "/home/kilian/vision-retrieval/models/"
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
    for run in range(5):
        for t in ["simple", "complex"]:
            for model in model_names:
                print(f"Running {model} on {t}...")
                embedd = embedding_func(MODEL_BASEPATH + model, BASE_MODEL)
                model_type = model.split("_")[0]
                results = run_eval(
                    f"/home/kilian/vision-retrieval/data/retrieval_queries/microtexts-retrieval-{t}/*.json",
                    f"/home/kilian/vision-retrieval/data/graphs/microtexts/*.json",
                    f"/home/kilian/vision-retrieval/evaluate_cbr/mac_{t}.json",
                    embedd,
                    lambda x: f"/home/kilian/vision-retrieval/data/eval_all/microtexts-retrieval-{t}/{model_type}/"
                    + x.split("/")[-1].split(".")[0]
                    + ".png",
                    lambda x: f"/home/kilian/vision-retrieval/data/eval_all/casebase/{model_type}/"
                    + x.split("/")[-1].split(".")[0]
                    + ".png",
                )
                res[model] = results
            df = pd.DataFrame(res)
            df.to_csv(f"results_{t}_{run}.csv")
