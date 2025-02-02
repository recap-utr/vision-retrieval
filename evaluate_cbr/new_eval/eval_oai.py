import json
from glob import glob
from PIL import Image
import base64
import io
from dotenv import load_dotenv
from openai import OpenAI
from correctness_completeness import _correctness_completeness_single
from ranx import Run, Qrels, evaluate
import statistics
from tqdm import tqdm
import copy
import os
import time

load_dotenv()  # take environment variables from .env.

client = OpenAI()
IMG_PLACEHOLDER = "data:image/png;base64,"
MODE = "complex"
BASEPATH = "/home/s4kibart/vision-retrieval/data/eval_all"
RANKING_SAVE_PATH = f"{BASEPATH}/results/oai/ranking_oai_epoch1_temp0_{MODE}.json"
RESULTS_PATH = f"{BASEPATH}/results/oai/results_oai_epoch1_temp0_{MODE}.json"
QUERY_IMAGES = f"{BASEPATH}/microtexts-retrieval-{MODE}/srip"
MAC_RESULTS_PATH = f"{BASEPATH}/mac_{MODE}.json"

OPENAI_MODEL = (
    "ft:gpt-4o-2024-08-06:wi2-trier-university:srip-900x2:APyQjzsR:ckpt-step-899"
)


queries_texts = (
    f"/home/s4kibart/vision-retrieval/data/retrieval_queries/microtexts-retrieval-{MODE}"
)

res = {}
ranking_oai = {}
qrels = {}
run = {}
sum_tokens = 0


def as_dict(queries, results, qrels, run, k):
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


def append_images(image) -> dict:
    # mac phase has this prefix, eval folder structure does not
    image = image.replace("microtexts/", "")
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_to_base64(Image.open(image).convert("RGB"))},
            }
        ],
    }


query_names = [
    path.split("/")[-1].split(".")[0] for path in glob(f"{queries_texts}/*.json")
]
loaded_existing = os.path.exists(RANKING_SAVE_PATH)
mac_results = build_mac_results(MAC_RESULTS_PATH)
print(f"Using these MAC results: {MAC_RESULTS_PATH}")
request_durations = []
if loaded_existing:
    with open(RANKING_SAVE_PATH, "r") as f:
        ranking_oai = json.load(f)
        print("Loaded existing ranking")
        sum_tokens = ranking_oai["sum_tokens"]
        request_durations = ranking_oai["request_durations"]
else:
    print(f"No existing ranking found. Starting new evaluation with {len(query_names)} {MODE} queries...")
for query_name in tqdm(query_names):
    query_text = f"{queries_texts}/{query_name}.json"
    query_image = f"{QUERY_IMAGES}/{query_name}.png"

    with open(query_text, "r") as f:
        query_text = json.load(f)
    reference_rankings = {
        k.split("/")[1]: 4 - v
        for k, v in query_text["userdata"]["cbrEvaluations"][0]["ranking"].items()
    }
    qrels[query_name] = reference_rankings
    casebase_path = f"{BASEPATH}/casebase/srip"
    casebase_images = [f"{casebase_path}/{n}.png" for n in mac_results[query_name]]
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
                "content": f"Take a look at the following images in space reclaiming icicle chart visualization. Image 1 represents the query, images 2-{1+len(reference_rankings.keys())} are retrieval candidates. Please rank all retrieval candidates (images 2-{1+len(reference_rankings.keys())}) in descending order based on their similarity to the query image. If there were, for example, the three retrieval candidates images 2-4, image 3 having the highest similarity, image 2 the second highest and image 4 the lowest, you would output: 3,2,4.",
            },
            {"role": "user", "content": "This is image 1, the query"},
        ]
        prompt_messages.append(append_images(query_image))
        prompt_messages.append(
            {
                "role": "user",
                "content": f"The following are images 2-{1+len(reference_rankings.keys())}, retrieval candidates:",
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
        request_start = time.time()
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=prompt_messages,
            temperature=0.0,
        )
        request_durations.append(time.time() - request_start)
        answer = completion.choices[0].message.content
        sum_tokens += completion.usage.total_tokens
    try:
        keys = [key.split("/")[-1] for key in mac_results[query_name]]
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
with open(RANKING_SAVE_PATH, "w") as f:
    json.dump(ranking_oai, f)
print("Starting evaluation")
start = time.time()
run = Run(run)
evaluate(
    Qrels(qrels),
    run,
    ["ndcg_burges", "ndcg", "map", "f1", "recall", "precision"],
    return_mean=False,
)
results = as_dict(query_names, copy.deepcopy(run.mean_scores), qrels, run, None)
results["sum_tokens"] = sum_tokens
results["request_duration"] = sum(request_durations)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f)
print(results)
print(f"Evaluation duration: {time.time() - start}")
