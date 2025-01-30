from logical import render
from pathlib import Path
import arguebuf as ab
from glob import glob
from tqdm import tqdm
import os

GRAPHS_PATH = "../data/graphs/microtexts"
OUTPUT_PATH = "../data/eval_all/casebase/logical"

os.makedirs(OUTPUT_PATH, exist_ok=True)

for graph in tqdm(glob(f"{GRAPHS_PATH}/*.json")):
    graph_name = Path(graph).stem
    graph = ab.load.file(graph)
    render(graph, Path(f"{OUTPUT_PATH}/{graph_name}.png"), normalize_graph=False)

