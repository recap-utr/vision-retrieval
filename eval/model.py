import arguebuf as ab
from PIL import Image
import torch
from typing import Callable
from pathlib import Path


class ImageGraph:
    def __init__(self, graph_path: str, image_path: str) -> None:
        self.graph = ab.load.file(graph_path)
        self.image = Image.open(image_path).convert("RGB")
        self.graph_path = graph_path
        self.image_path = image_path
        self.name = graph_path.split("/")[-1].split(".")[0].lower()


class ImageEmbeddingGraph(ImageGraph):
    def __init__(
        self,
        graph_path: Path,
        image_path: Path,
        embedding_func: Callable[..., torch.Tensor] | None = None,
        name: str | None = None,
    ):
        self.graph = ab.load.file(graph_path)
        self.image = Image.open(image_path).convert("RGB")
        self.name = graph_path.stem
        self.graph_path = graph_path
        self.image_path = image_path
        self.embedding = embedding_func(self.image)
