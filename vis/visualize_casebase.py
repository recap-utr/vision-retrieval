from logical import render
from pathlib import Path
import arguebuf as ab
from glob import glob
from tqdm import tqdm
import os

BASEPATH_GRAPHS = "../data/retrieval_queries"
BASEPATH_IMAGES = "../data/eval_all"

for t in ("simple", "complex"):
    graphs_path = f"{BASEPATH_GRAPHS}/microtexts-retrieval-{t}"
    output_path = f"{BASEPATH_IMAGES}/microtexts-retrieval-{t}/logical"

    os.makedirs(output_path, exist_ok=True)

    for graph in tqdm(glob(f"{graphs_path}/*.json")):
        graph_name = Path(graph).stem
        graph = ab.load.file(graph)
        render(graph, Path(f"{output_path}/{graph_name}.png"), normalize_graph=False)

