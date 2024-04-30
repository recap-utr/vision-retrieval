# import faiss
import torch
from glob import glob
from PIL import Image
import faiss
from .model import Model
from .visualize import export_graph
import arguebuf as ab
import uuid
from .mac import retrieve_mac, retrieve_fac
from torch.nn.functional import cosine_similarity
from time import time
from .text_embeddings import embedd_texts
import torch
from tqdm import tqdm

EXPORT_PATH = "/tmp/"

def get_text(graph: ab.Graph):
    texts = [node.label for node in graph.nodes.values() if node.label != "Support" and node.label != "Attack"]
    return " ".join(texts)

class Casebase:
    def __init__(self, folder: str, image_folder: str, model_name="", model=None):
        self.folder = folder
        self.image_folder = image_folder
        if model:
            self.model = model
        else:
            self.model = Model(model_name)
        
        text_files = glob(f"{folder}/*.json")
        self.text_cb: dict[str, torch.Tensor] = {}
        start = time()
        texts = {}
        for f in tqdm(text_files):
            graph_id = f.split("/")[-1].split(".")[0]
            graph = ab.load.file(f)
            text = " ".join([node.label for node in graph.nodes.values() if node.label != "Support" and node.label != "Attack"])
            texts[graph_id] = text
        embeddings = embedd_texts(list(texts.values()))
        for i, graph_id in enumerate(texts.keys()):
            self.text_cb[graph_id] = embeddings[i]
        print(f"Text cb took {time() - start} seconds")

        files = glob(f"{image_folder}/*.png")
        print(f"Loading {len(files)} images into memory...")
        start = time()

        images = [Image.open(f).convert("RGB") for f in files]
        images = [self.model.processor(image, return_tensors="pt")["pixel_values"] for image in images]
        images = torch.cat(images, dim=0)
        print(images.shape)

        with torch.no_grad():
            embeddings = self.model(images)
        self.image_cb: dict[str, torch.Tensor] = {f.split("/")[-1].split(".")[0]: embeddings[i] for i, f in enumerate(files)}
        print(f"Image cb took {time() - start} seconds")

    def retrieve_mac(self, query: ab.Graph, k: int = 5) -> dict[str, float]:
        query_embedding = embedd_texts([get_text(query)])[0]
        sims = {graph_id: cosine_similarity(torch.from_numpy(self.text_cb[graph_id]), torch.from_numpy(query_embedding), dim=0) for graph_id, query_embedding in self.text_cb.items()}
        # sort by similarity desc
        sorted_similarities = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        # transform to map: graph_id -> similarity
        return dict(sorted_similarities[:k])
    def query_casebase(self, query: ab.Graph, k: int = 5, mac: bool = False) -> dict[str, float]:
        query_id = str(uuid.uuid4())
        path = f"{EXPORT_PATH}{query_id}.png"
        export_graph(query, path)
        image = Image.open(path).convert("RGB")
        image = self.model.processor(image, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            embedding = self.model(image).squeeze()
        if not mac:
            sims = {graph_id: cosine_similarity(embedding, graph_embedding, dim=0) for graph_id, graph_embedding in self.image_cb.items()}
            # sort by similarity desc
            sorted_similarities = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_similarities[:k])
        similarities = {}
        start = time()
        mac_ids = self.retrieve_mac(query, k*2)
        print("MAC took: ", time() - start)
        # for graph_id in mac_ids: for Vorfilterung durch mac
        for graph_id, sim in mac_ids.items():
            graph_embedding = self.image_cb[graph_id]
            similarities[graph_id] = (cosine_similarity(embedding, graph_embedding, dim=0).item() + sim)/2
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_similarities[:k])
    
    def query_fac(self, query: ab.Graph, k: int = 5) -> dict[str, float]:
        mac_ids = retrieve_fac(self.folder, self.retrieve_mac(query, k*2), query, k)
        sorted_similarities = sorted(mac_ids.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_similarities[:k])

# class Casebase_FAISS:
#     def __init__(self, folder: str, image_folder: str):
#         d = 768 # dimension
#         self.index = faiss.IndexFlatIP(d)
#         self.cb: dict[int, str] = {}
#         self.reverse_cb: dict[str, int] = {}
#         self.folder = folder
#         self.image_folder = image_folder

#         files = glob(f"{image_folder}/*.png")
#         counter = 0
#         images = [Image.open(f).convert("RGB") for f in files]

#         with torch.no_grad():
#             for f in files:
#                 graph_id = f.split("/")[-1].split(".")[0]
#                 self.cb[counter] = graph_id
#                 self.reverse_cb[graph_id] = counter
#                 counter += 1
#             embeddings = embedd_many(images)
#         self.index.add(embeddings)

#     def query_casebase(self, query: ab.Graph, k: int = 5, mac: bool = False) -> dict[str, float]:
#         query_id = str(uuid.uuid4())
#         path = f"{EXPORT_PATH}{query_id}.png"
#         export_graph(query, path)
#         image = Image.open(path).convert("RGB")
#         with torch.no_grad():
#             embedding = embedd_many(images=[image])
#         if not mac:
#             distances, indices = self.index.search(embedding, k)
#             distances = distances.squeeze()
#             indices = indices.squeeze()
#             ret = {self.cb[indices[i]]: distances[i] for i in range(len(indices))}
#             return ret
#         mac_results = retrieve_mac(self.folder, query)
#         sel = faiss.IDSelectorBatch([self.reverse_cb[graph_id] for graph_id in mac_results.keys()])
#         params = faiss.SearchParameters()
#         params.sel = sel
#         distances, indices = self.index.search(embedding, k, params = params)
#         distances = distances.squeeze()
#         indices = indices.squeeze()
#         ret = {self.cb[indices[i]]: distances[i] for i in range(len(indices))}
#         return ret
