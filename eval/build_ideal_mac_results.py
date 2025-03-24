import json
from glob import glob
from tqdm import tqdm
from pathlib import Path


def convert(basepath: Path, mode: str, outpath: Path):
    """
    Writes the ideal MAC results for every arguement retrieval query of a given mode in a given basepath to a single json file.
    :param basepath: Path to the directory containing the microtexts-retrieval-{mode} directories.
    """
    res = {}
    for f in tqdm(glob(f"{basepath}/microtexts-retrieval-{mode}/*.json")):
        queryname = f.split("/")[-1].split(".")[0]
        with open(f, "r") as f:
            data = json.load(f)
            res[queryname] = list(
                data["userdata"]["cbrEvaluations"][0]["ranking"].keys()
            )
    with open(outpath, "w") as f:
        json.dump(res, f, indent=4)
        print(f"Saved {outpath}")
