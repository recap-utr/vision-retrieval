from uuid import uuid4
from random import randint
import arguebuf as ab
from pathlib import Path
from os import makedirs
import random
from render import render, RenderMethod
import typer
from multiprocessing import Pool
from enum import Enum
from time import time


class NormalizeHeight(str, Enum):
    TRUE = "true"
    FALSE = "false"
    RANDOM = "random"


app = typer.Typer()


def generate_tree(max_depth: int, max_branching_degree: int, depth: int = 0):
    if depth >= max_depth:
        return {"id": str(uuid4()), "children": []}
    return {
        "id": str(uuid4()),
        "children": [
            generate_tree(max_depth, max_branching_degree, depth + 1)
            for _ in range(randint(0, max(2, max_branching_degree - depth)))
        ],
    }


def convert_tree_to_graph(
    roots: list[dict], depth=0, graph=None
) -> tuple[ab.AbstractNode, ab.Graph]:
    graph = ab.Graph() if graph is None else graph
    for root in roots:
        choice = random.randint(0, 2)
        if depth == 0 or choice == 0:
            root_node = ab.AtomNode("claim")
        elif choice == 1:
            root_node = ab.SchemeNode(ab.Attack.DEFAULT)
        else:
            root_node = ab.SchemeNode(ab.Support.DEFAULT)
        graph.add_node(root_node)
        children = root["children"] if isinstance(root, dict) else root.children
        for child in children:
            child_node, _ = convert_tree_to_graph([child], depth=depth + 1, graph=graph)
            graph.add_edge(ab.Edge(child_node, root_node))
    return root_node, graph


@app.command()
def generate_random_graphs_command(
    save_path: Path,
    thread_count: int = 1,
    graphs_per_thread: int = 100_000,
    max_depth: int = 6,
    max_branching_degree: int = 5,
    method: RenderMethod = RenderMethod.SRIP2,
    max_tree_roots: int = 1,
    normalize_height: NormalizeHeight = NormalizeHeight.RANDOM,
):
    makedirs(save_path, exist_ok=True)
    with Pool(thread_count) as p:
        p.starmap(
            _generation_workload,
            (
                (
                    save_path,
                    i,
                    graphs_per_thread,
                    max_depth,
                    max_branching_degree,
                    method,
                    max_tree_roots,
                    normalize_height,
                )
                for i in range(thread_count)
            ),
        )


def _generation_workload(
    save_path: Path,
    thread_num: int,
    graphs_per_thread: int = 100_000,
    max_depth: int = 9,
    max_branching_degree: int = 7,
    method: RenderMethod = RenderMethod.SRIP2,
    max_tree_roots: int = 1,
    normalize_height: NormalizeHeight = NormalizeHeight.RANDOM,
):
    start = time()
    for i in range(graphs_per_thread):
        if i % 1000 == 0:
            print(
                f"Thread {thread_num} is at {i} graphs after {time() - start} seconds"
            )
        md = randint(1, max_depth)
        trees = []
        for _ in range(randint(1, max_tree_roots)):
            trees.append(generate_tree(md, max_branching_degree))
        graph = convert_tree_to_graph(trees)[1]
        path = Path(f"{save_path}/{thread_num}_{i}.png")
        norm_height = False if normalize_height == NormalizeHeight.FALSE else True
        if normalize_height == NormalizeHeight.RANDOM:
            norm_height = random.choice([True, False])
        try:
            render(graph, path, method=method, normalize_height=norm_height)
        except Exception as e:
            print(f"Error rendering graph: {e}")


if __name__ == "__main__":
    app()
