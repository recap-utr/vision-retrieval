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

BASEPATH = "/home/s4kibart/vision-retrieval/data/eval_all"
RANKING_SAVE_PATH = f"{BASEPATH}/ranking_oai_epoch2_simple.json"
RESULTS_PATH = f"{BASEPATH}/results_oai_epoch2_simple.json"
QUERY_IMAGES = f"{BASEPATH}/microtexts-retrieval-simple/srip"

OPENAI_MODEL = (
    "ft:gpt-4o-2024-08-06:wi2-trier-university:srip-900x2:APyQkjBm:ckpt-step-1798"
)


queries_texts = (
    "/home/s4kibart/vision-retrieval/data/retrieval_queries/microtexts-retrieval-simple"
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


def append_images(image) -> dict:
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
if loaded_existing:
    with open(RANKING_SAVE_PATH, "r") as f:
        ranking_oai = json.load(f)
        print("Loaded existing ranking")
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
    casebase_images = [f"{casebase_path}/{n}.png" for n in reference_rankings.keys()]
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

        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=prompt_messages,
        )
        answer = completion.choices[0].message.content
        sum_tokens += completion.usage.total_tokens
    order = [int(i) for i in answer.split(",")]
    keys = list(reference_rankings.keys())
    # assert len(set(order)) == len(
    #     reference_rankings
    # ), "Rankings must be unique and equal to the number of images"
    # assert set(order) == set(
    #     range(2, len(reference_rankings) + 2)
    # ), "Rankings must be a permutation of 1 to N"
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
ranking_oai["sum_tokens"] = sum_tokens
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
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f)
print(results)
print(f"Duration: {time.time() - start}")
print(f"Sum tokens: {sum_tokens}")
