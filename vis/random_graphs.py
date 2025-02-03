from uuid import uuid4
from random import randint
import os
import arguebuf as ab
from logical import render
from pathlib import Path
from logical import NodeWrapper, render
from srip import convert_from_AbstractNode_to_Node, SRIP2, SRIP_Config, default_weight
from treemaps import visualize_treemap_inmem, standard_resize
from tqdm import tqdm
from os import makedirs
import random

MAX_DEPTH = 9
MAX_BRANCHING_DEGREE = 7


def parse_args():
    parser = argparse.ArgumentParser(description="Generate random graphs.")
    parser.add_argument(
        "--max_depth", type=int, default=9, help="Maximum depth of the tree."
    )
    parser.add_argument(
        "--max_branching_degree",
        type=int,
        default=7,
        help="Maximum branching degree of the tree.",
    )
    parser.add_argument(
        "--visualization",
        type=str,
        choices=["logical", "srip", "treemap"],
        default="logical",
        help="Type of visualization.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="random_logical_srip/logical/images/",
        help="Path to save the generated images.",
    )
    return parser.parse_args()


args = parse_args()
MAX_DEPTH = args.max_depth
MAX_BRANCHING_DEGREE = args.max_branching_degree
VISUALIZATION = args.visualization

WIDTH = 256
HEIGHT = 256
os.chdir("../data/")
savepath = args.save_path
makedirs(savepath, exist_ok=True)


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


def generate_random_graphs(thread_num: int, k: int = 100_000):
    for i in tqdm(range(k)):
        md = randint(1, MAX_DEPTH)
        tree = generate_tree(md)
        graph = convert_tree_to_graph(tree)[1]
        path = Path(f"{savepath}{thread_num}_{i}.png")
        if VISUALIZATION == "logical":
            render(graph, path, normalize_graph=False)
        elif VISUALIZATION == "srip":
            root, graph = convert_tree_to_graph(tree)
            root = convert_from_AbstractNode_to_Node(graph, root)
            config = SRIP_Config()
            config.gamma = random.random() / 10
            config.rho = random.random()
            config.epsilon = random.random() * WIDTH
            config.sigma = random.random()
            config.lambda_ = randint(1, 5)
        else:
            visualize_treemap_inmem(graph, path, HEIGHT, WIDTH)
            standard_resize(path)

        # print(f"Generated {i+1} graphs")


from multiprocessing import Pool
import argparse

NUM_PROCESSES = os.cpu_count()

with Pool(NUM_PROCESSES) as p:
    p.map(generate_random_graphs, range(NUM_PROCESSES))
