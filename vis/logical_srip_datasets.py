from logical import NodeWrapper, render
from srip import SRIP2, convert_from_AbstractNode_to_Node, default_weight
from util import find_heighest_root_node
import arguebuf as ab
from glob import glob
from tqdm import tqdm
from pathlib import Path
import os
from multiprocessing import Pool

# folgende problematisch (stuck an 241/16)
# "kialo": "txt",
# "kialo-nilesc": "txt",

dataset_dict = {
    "araucaria": "json",
    "iac": "json",
    "kialo-graphnli": "json",
    "persuasive-essays": "ann",
    "qt30": "json",
    "us-2016": "json",
    "microtexts-v2": "xml",
}


base_path = "../data"
target_dir = f"{base_path}/arg_finetune_logical_srip"
srip_dir = f"{target_dir}/srip2/train"
logical_dir = f"{target_dir}/logical/train"
os.makedirs(srip_dir, exist_ok=True)
os.makedirs(logical_dir, exist_ok=True)


def process_dataset(name, ext):
    for file in tqdm(glob(f"{base_path}/graphs/{name}/*.{ext}")):
        try:
            graph = ab.load.file(file)
            mj = find_heighest_root_node(graph)
            root_srip = convert_from_AbstractNode_to_Node(graph, mj)
            path = Path(f"{srip_dir}/{name}-{file.split('/')[-1].replace(ext, 'png')}")
            SRIP2(root_srip, graph, path, default_weight)
            path = Path(
                f"{logical_dir}/{name}-{file.split('/')[-1].replace(ext, 'png')}"
            )
            render(graph, path)
        except Exception as e:
            print("error processing", file)
            print(e)
            continue
    print(f"Finished processing {name}")


if __name__ == "__main__":
    with Pool(len(dataset_dict)) as p:
        p.starmap(process_dataset, dataset_dict.items())

# eval casebase + queries
# casebase = (
#     f"{base_path}/graphs/microtexts/*.json",
#     f"{base_path}/eval_all/casebase/logical",
# )
# retrieval_simple = (
#     f"{base_path}/retrieval_queries/microtexts-retrieval-simple/*.json",
#     f"{base_path}/eval_all/microtexts-retrieval-simple/logical",
# )
# retrieval_complex = (
#     f"{base_path}/retrieval_queries/microtexts-retrieval-complex/*.json",
#     f"{base_path}/eval_all/microtexts-retrieval-complex/logical",
# )
# generation_tasks_logical = [casebase, retrieval_simple, retrieval_complex]

# for source, target in generation_tasks_logical:
#     os.makedirs(target, exist_ok=True)
#     for file in tqdm(glob(source)):
#         try:
#             graph = ab.load.file(file)
#             mj = find_major_claim(graph)
#             path = Path(f"{target}/{file.split('/')[-1].replace('json', 'png')}")
#             render(graph, path)
#         except Exception as e:
#             print("error processing", file)
#             print(e)
#             continue

# casebase = (
#     f"{base_path}/graphs/microtexts/*.json",
#     f"{base_path}/eval_all/casebase/srip",
# )
# retrieval_simple = (
#     f"{base_path}/retrieval_queries/microtexts-retrieval-simple/*.json",
#     f"{base_path}/eval_all/microtexts-retrieval-simple/srip",
# )
# retrieval_complex = (
#     f"{base_path}/retrieval_queries/microtexts-retrieval-complex/*.json",
#     f"{base_path}/eval_all/microtexts-retrieval-complex/srip",
# )
# generation_tasks_srip = [casebase, retrieval_simple, retrieval_complex]

# for source, target in generation_tasks_srip:
#     os.makedirs(target, exist_ok=True)
#     for file in tqdm(glob(source)):
#         try:
#             graph = ab.load.file(file)
#             mj = find_major_claim(graph)
#             path = Path(f"{target}/{file.split('/')[-1].replace('json', 'png')}")
#             SRIP2(
#                 convert_from_AbstractNode_to_Node(graph, mj),
#                 graph,
#                 path,
#                 default_weight,
#             )
#         except Exception as e:
#             print("error processing", file)
#             print(e)
#             continue
