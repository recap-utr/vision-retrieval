from uuid import uuid4
from random import randint
import os

MAX_DEPTH = 9
MAX_BRANCHING_DEGREE = 7


def generate_tree(max_depth, depth: int = 0):
    if depth >= max_depth:
        return {
            "id": str(uuid4()),
            "children": [],
            "label": "Attack" if randint(0, 1) == 0 else "Support",
        }
    return {
        "id": str(uuid4()),
        "children": [
            generate_tree(max_depth, depth + 1)
            for _ in range(randint(0, max(2, MAX_BRANCHING_DEGREE - depth)))
        ],
        "label": "Attack" if randint(0, 1) == 0 else "Support",
    }


# generate random graphs
from make_treemap_Snodes import get_treemap_rects, draw_treemap, standard_resize
from tqdm import tqdm

width = 256
height = 256
savepath = "random_treemaps"
os.makedirs(savepath, exist_ok=True)


def generate_random_graphs(thread_num: int, k: int = 100_000):
    for i in tqdm(range(k)):
        md = randint(1, MAX_DEPTH)
        tree = generate_tree(md)
        rects = get_treemap_rects(tree, 0, 0, width, height, True)
        path = f"{savepath}/{thread_num}-{i}.png"
        draw_treemap(rects, height, width, path)
        standard_resize(path)
        # print(f"Generated {i+1} graphs")


from multiprocessing import Pool

NUM_PROCESSES = os.cpu_count()

with Pool(NUM_PROCESSES) as p:
    p.map(generate_random_graphs, range(NUM_PROCESSES))
