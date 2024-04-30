from casebase import Casebase
import arguebuf as ab
from time import time
import pickle
import os
from torch.nn.functional import cosine_similarity
import torch
if __name__ == "__main__":
    # generate cb if not exists
    if os.path.exists("cb.pickle"):
        with open("cb.pickle", "rb") as f:
            cb = pickle.load(f)
    else:
        cb = Casebase(image_folder="/home/kilian/ba/data/kialo-graphnli-images", folder="/home/kilian/ba/data/kialo-graphnli")
        # save cb
        with open("cb.pickle", "wb") as f:
            pickle.dump(cb, f)

    query = ab.load.file("/home/kilian/ba/data/kialo-graphnli/333.json")
    start = time()
    print(cb.query_casebase(query, mac=False, k=30))
    print("Image based: ", time() - start)
    start = time()
    print(cb.query_casebase(query, mac=True, k=30))
    print("Image based (mac): ", time() - start)
    start = time()
    print(cb.query_fac(query, k=30))
    print("fac :", time() - start)