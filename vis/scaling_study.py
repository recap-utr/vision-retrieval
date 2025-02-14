from glob import glob
from tqdm import tqdm
from srip import convert_from_AbstractNode_to_Node, SRIP2, SRIP_Config, default_weight
import arguebuf as ab
from pathlib import Path
from util import find_major_claim
from time import time
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
import torch
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def embedding_func(model_path: str, base_model: str):
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


REQUESTS_GRAPHS = "../data/scaling_requests/*.json"
CB = "../data/scaling_cb.json"
TIMES = 10
DEST = "../data/retrieval_scaling/queries"
CASEDEST = "../data/retrieval_scaling/cb/case.png"
BASE_MODEL = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft"


visualization_times = {k.split("/")[-1].split(".")[0]: 0 for k in glob(REQUESTS_GRAPHS)}
embedding_times = {k.split("/")[-1].split(".")[0]: 0 for k in glob(REQUESTS_GRAPHS)}
similarity_computation_times = {
    k.split("/")[-1].split(".")[0]: 0 for k in glob(REQUESTS_GRAPHS)
}
similarity_computation_times2 = {
    k.split("/")[-1].split(".")[0]: 0 for k in glob(REQUESTS_GRAPHS)
}
embeddings = {}


config = SRIP_Config()
emb = embedding_func("../vis_models/srip_ft_arg", BASE_MODEL)


print("measuring visualization times")
for i in tqdm(range(TIMES)):
    for file in glob(REQUESTS_GRAPHS):
        graph = ab.load.file(file)
        number_s_nodes = file.split("/")[-1].split(".")[0]
        path = Path(f"{DEST}/{file.split('/')[-1].replace('json', 'png')}")
        start = time()
        mj = find_major_claim(graph)
        root_srip = convert_from_AbstractNode_to_Node(graph, mj)
        root = convert_from_AbstractNode_to_Node(graph, mj)
        SRIP2(root, graph, path, default_weight)
        visualization_times[number_s_nodes] += time() - start

# visualize case
case_graph = ab.load.file(CB)
case_mj = find_major_claim(case_graph)
case_path = Path(CASEDEST)
SRIP2(
    convert_from_AbstractNode_to_Node(case_graph, case_mj),
    case_graph,
    case_path,
    default_weight,
)

print("measuring embedding times")
for i in tqdm(range(TIMES)):
    local_times = {}

    for file in glob(f"{DEST}/*.png"):
        image = Image.open(file).convert("RGB")
        number_s_nodes = file.split("/")[-1].split(".")[0]
        start = time()
        embeddings[number_s_nodes] = emb(image)
        embedding_times[number_s_nodes] += time() - start

# embedd case
case_image = Image.open(CASEDEST).convert("RGB")
case_emb = emb(case_image)

print("measuring similarity computation times: cosine")
for i in tqdm(range(TIMES)):
    local_times = {}

    for number_s_nodes, emb in embeddings.items():
        start = time()
        torch.nn.functional.cosine_similarity(emb, case_emb)
        similarity_computation_times[number_s_nodes] += time() - start

visualization_times = {k: v / TIMES for k, v in visualization_times.items()}
embedding_times = {k: v / TIMES for k, v in embedding_times.items()}
similarity_computation_times = {
    k: v / TIMES for k, v in similarity_computation_times.items()
}

print("measuring similarity computation times: dot")
for i in tqdm(range(TIMES)):
    local_times = {}

    for number_s_nodes, emb in embeddings.items():
        start = time()
        torch.dot(emb.flatten(), case_emb.flatten())
        similarity_computation_times2[number_s_nodes] += time() - start

visualization_times = {k: v / TIMES for k, v in visualization_times.items()}
embedding_times = {k: v / TIMES for k, v in embedding_times.items()}
similarity_computation_times2 = {
    k: v / TIMES for k, v in similarity_computation_times.items()
}

print("visualization times")
print(visualization_times)
print("embedding times")
print(embedding_times)
print("sim times: cosine")
print(similarity_computation_times)
print("sim times: dot")
print(similarity_computation_times2)
with open("../data/retrieval_scaling/scale_results.json", "w") as f:
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
