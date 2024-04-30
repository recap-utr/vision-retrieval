import arguebuf as ab
from PIL import Image
import torch
from typing import Callable
from copy import deepcopy

class ImageGraph:
    def __init__(self, graph_path: str, image_path: str) -> None:
        self.graph = ab.load.file(graph_path)
        self.image = Image.open(image_path).convert("RGB")
        self.graph_path = graph_path
        self.image_path = image_path
        self.name = graph_path.split("/")[-1].split(".")[0].lower()

class ImageEmbeddingGraph(ImageGraph):
    def __init__ (self, image_graph: ImageGraph, embedding_func: Callable[..., torch.Tensor] | None = None):
        self.graph = ab.load.file(image_graph.graph_path)
        self.image = Image.open(image_graph.image_path).convert("RGB")
        self.name = image_graph.name
        self.graph_path = image_graph.graph_path
        self.image_path = image_graph.image_path
        self.embedding = embedding_func(self.image)