from uuid import uuid4
from random import randint
import os
import arguebuf as ab
from logical import render
from pathlib import Path

MAX_DEPTH = 9
MAX_BRANCHING_DEGREE = 7


def generate_tree(max_depth, depth: int = 0):
    if depth >= max_depth:
        return {"id": str(uuid4()), "children": []}
    return {
        "id": str(uuid4()),
        "children": [
            generate_tree(max_depth, depth + 1)
            for _ in range(randint(0, max(2, MAX_BRANCHING_DEGREE - depth)))
        ],
    }


def convert_tree_to_graph(
    root, depth=0, graph=None
) -> tuple[ab.AbstractNode, ab.Graph]:
    root_node = (
        ab.AtomNode("major_claim")
        if depth == 0
        else ab.SchemeNode(
            ab.Attack.DEFAULT if randint(0, 1) == 0 else ab.Support.DEFAULT
        )
    )
    graph = ab.Graph() if graph is None else graph
    graph.add_node(root_node)
    children = root["children"] if isinstance(root, dict) else root.children
    for child in children:
        child_node, _ = convert_tree_to_graph(child, depth=depth + 1, graph=graph)
        graph.add_edge(ab.Edge(child_node, root_node))
    return root_node, graph


# generate random graphs
from logical import NodeWrapper, render
from tqdm import tqdm
from os import makedirs

width = 256
height = 256
os.chdir("../data/")
savepath = "random_logical_srip/logical/images/"
makedirs(savepath, exist_ok=True)


def generate_random_graphs(thread_num: int, k: int = 100_000):
    for i in tqdm(range(k)):
        md = randint(1, MAX_DEPTH)
        tree = generate_tree(md)
        graph = convert_tree_to_graph(tree)[1]
        path = Path(f"{savepath}{thread_num}_{i}.png")
        render(graph, path, normalize_graph=False)
        # print(f"Generated {i+1} graphs")


from multiprocessing import Pool

NUM_PROCESSES = os.cpu_count()

with Pool(NUM_PROCESSES) as p:
    p.map(generate_random_graphs, range(NUM_PROCESSES))
