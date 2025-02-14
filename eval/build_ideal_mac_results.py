import json
from glob import glob
from tqdm import tqdm

BASEPATH = "../data/retrieval_queries"
MODES = ["simple", "complex"]
OUTPUT_PATH = "../data/eval_all"

for mode in MODES:
    res = {}
    for f in tqdm(glob(f"{BASEPATH}/microtexts-retrieval-{mode}/*.json")):
        queryname = f.split("/")[-1].split(".")[0]
        with open(f, "r") as f:
            data = json.load(f)
            res[queryname] = list(
                data["userdata"]["cbrEvaluations"][0]["ranking"].keys()
            )
    with open(f"{OUTPUT_PATH}/ideal_mac_{mode}.json", "w") as f:
        json.dump(res, f, indent=4)
        print(f"Saved {OUTPUT_PATH}/ideal_mac_{mode}.json")
