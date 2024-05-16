from logical import NodeWrapper, render
from srip import SRIP2, convert_from_AbstractNode_to_Node, default_weight
from util import find_major_claim
import arguebuf as ab
from glob import glob
from tqdm import tqdm
from pathlib import Path
import os

dataset_dict = {
    "araucaria": "json",
    "iac": "json",
    "kialo-graphnli": "json",
    "microtexts": "json",
    "persuasive-essays": "ann",
    "qt30": "json",
    "us-2016": "json",
}


base_path = "../data"
target_dir = f"{base_path}/pretrain-logical-srip"
srip_dir = f"{target_dir}/srip2"
logical_dir = f"{target_dir}/logical"
os.makedirs(srip_dir, exist_ok=True)
os.makedirs(logical_dir, exist_ok=True)


# for name, ext in tqdm(dataset_dict.items()):
#     for file in glob(f"{base_path}/graphs/{name}/*.{ext}"):
#         try:
#             graph = ab.load.file(file)
#             mj = find_major_claim(graph)
#             root_srip = convert_from_AbstractNode_to_Node(graph, mj)
#             path = Path(f"{srip_dir}/{name}-{file.split('/')[-1].replace(ext, 'png')}")
#             SRIP2(root_srip, graph, path, default_weight)
#             path = Path(
#                 f"{logical_dir}/{name}-{file.split('/')[-1].replace(ext, 'png')}"
#             )
#             render(graph, path)
#         except Exception as e:
#             print("error processing", file)
#             print(e)
#             continue

# eval casebase + queries
casebase = (
    f"{base_path}/graphs/microtexts/*.json",
    f"{base_path}/eval_all/casebase/logical",
)
retrieval_simple = (
    f"{base_path}/retrieval_queries/microtexts-retrieval-simple/*.json",
    f"{base_path}/eval_all/microtexts-retrieval-simple/logical",
)
retrieval_complex = (
    f"{base_path}/retrieval_queries/microtexts-retrieval-complex/*.json",
    f"{base_path}/eval_all/microtexts-retrieval-complex/logical",
)
generation_tasks_logical = [casebase, retrieval_simple, retrieval_complex]

for source, target in generation_tasks_logical:
    os.makedirs(target, exist_ok=True)
    for file in tqdm(glob(source)):
        try:
            graph = ab.load.file(file)
            mj = find_major_claim(graph)
            path = Path(f"{target}/{file.split('/')[-1].replace('json', 'png')}")
            render(graph, path)
        except Exception as e:
            print("error processing", file)
            print(e)
            continue

casebase = (
    f"{base_path}/graphs/microtexts/*.json",
    f"{base_path}/eval_all/casebase/srip",
)
retrieval_simple = (
    f"{base_path}/retrieval_queries/microtexts-retrieval-simple/*.json",
    f"{base_path}/eval_all/microtexts-retrieval-simple/srip",
)
retrieval_complex = (
    f"{base_path}/retrieval_queries/microtexts-retrieval-complex/*.json",
    f"{base_path}/eval_all/microtexts-retrieval-complex/srip",
)
generation_tasks_srip = [casebase, retrieval_simple, retrieval_complex]

for source, target in generation_tasks_srip:
    os.makedirs(target, exist_ok=True)
    for file in tqdm(glob(source)):
        try:
            graph = ab.load.file(file)
            mj = find_major_claim(graph)
            path = Path(f"{target}/{file.split('/')[-1].replace('json', 'png')}")
            SRIP2(
                convert_from_AbstractNode_to_Node(graph, mj),
                graph,
                path,
                default_weight,
            )
        except Exception as e:
            print("error processing", file)
            print(e)
            continue
