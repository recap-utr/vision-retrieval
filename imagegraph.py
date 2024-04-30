import torch
import wandb

BASE_PATH = "/home/s4kibart/ba/data"
MODEL_PATH = "facebook/dinov2-base"
MODEL_NAME = MODEL_PATH.split("/")[-1]

config = {
    "epochs": 50,
    "positives": 15,
    "negatives": 100,
    "hard_negatives": 12,
    "batch_size": 1,
    "lr": 0.03,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "gamma": 0.5,
    "step_size": 20,
    "temperature": 0.1,
    "optimizer": "sgd",
    # ------------------
    "model": MODEL_PATH,
}

wandb.init(project="imagegraph_finetuning", notes=MODEL_NAME, config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from evaluate_cbr.visualize import export_graph
import uuid
import arguebuf as ab
from PIL import Image
import os

class ImageGraph:    
    def __init__(self, path, image_path=None) -> None:
        try:
            self.path = path
            self.graph: ab.Graph = ab.load.file(path)
            if image_path is None:
                self.image_path = "/tmp" + str(uuid.uuid4()) + ".png"
                export_graph(self.graph, self.image_path)
            else:
                self.image_path = image_path
                # raise exception if image_path is not valid
                if not os.path.exists(self.image_path):
                    raise Exception("Image path does not exist")                  
            self.image = Image.open(self.image_path).convert("RGB")
        except Exception as e:
            self.graph = None


import math

def heuristic_distance(g1: ImageGraph, g2: ImageGraph) -> float:
    graph1 = g1.graph
    graph2 = g2.graph
    no_edges = len(graph1.edges)
    no_i_nodes = len(graph1.atom_nodes)
    no_s_nodes = len(graph1.scheme_nodes)
    no_support = len([n for n in graph1.scheme_nodes.values() if n.label == "Support"])

    # calculate statistics for graph 2
    no_edges2 = len(graph2.edges)
    no_i_nodes2 = len(graph2.atom_nodes)
    no_s_nodes2 = len(graph2.scheme_nodes)
    no_support2 = len([n for n in graph2.scheme_nodes.values() if n.label == "Support"])


    # calculate normalized deltas
    delta_edges = (no_edges - no_edges2)
    delta_i_nodes = (no_i_nodes - no_i_nodes2)
    delta_s_nodes = (no_s_nodes - no_s_nodes2)
    delta_depth = (g1.image.height // 28 - g2.image.height // 28)
    delta_support = (no_support - no_support2)



    # calculate euclidean distance
    return math.sqrt(delta_edges ** 2 + delta_i_nodes ** 2 + delta_s_nodes ** 2 + delta_depth ** 2 + delta_support ** 2)

import torch
from torchvision.transforms import v2

transforms = v2.Compose([
    # v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    # v2.RandomRotation(45, fill=1),
    v2.RandomAffine(degrees=(-179, 180), translate=(0.1, 0.3), scale=(0.5, 0.75), fill=1),
    # v2.RandomResizedCrop((224, 224), antialias=True),
    # v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    v2.RandomZoomOut(fill=1),
    v2.Resize((224, 224), antialias=True),
    v2.ElasticTransform(fill=1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

from torch.utils.data import Dataset

class ImageGraphDataset:

    def __init__(self, paths: list[str], image_paths: list[str]) -> None:
        self.graphs: list[ImageGraph] = []
        for path, image_path in zip(paths, image_paths):
            g = ImageGraph(path, image_path)
            if g.graph is not None:
                self.graphs.append(g)

    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, index: int) -> ImageGraph:
        return self.graphs[index]
    
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

from glob import glob
from time import time

start = time()

# remove microtexts from finetuning data

graph_paths = {"araucaria": "json", "iac": "json", "kialo-graphnli": "json",  "persuasive-essays": "ann", "qt30": "json", "us-2016": "json"}
paths_dict = {k: glob(f"{BASE_PATH}/{k}/*.{v}") for k, v in graph_paths.items()}
# sort paths
for k, v in paths_dict.items():
    paths_dict[k] = sorted(v)

# use paths_dict to generate image_paths_dict
image_paths_dict = {}
for k, v in paths_dict.items():
    image_paths_dict[k] = [path.replace("json", "png").replace(k, f"{k}-images") for path in v]

datasets = {k: ImageGraphDataset(v, image_paths_dict[k]) for k, v in paths_dict.items()}
print(f"Loading took {time() - start} seconds")

import torch
from transformers import AutoModel, AutoImageProcessor
import numpy as np

model = AutoModel.from_pretrained(MODEL_PATH)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

class ImageTriplesDataset(Dataset):

    def __init__(self, datasets: dict[str, ImageGraphDataset], num_pos: int, num_neg: int, num_hard_neg: int) -> None:
        assert num_neg > num_hard_neg
        self.datasets = datasets
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.num_hard_neg = num_hard_neg
        self.current_dataset = 0
        self.datasets_names = [d for d in datasets.keys()]
        self.lengths = [len(d) for d in datasets.values()]

    def __len__(self) -> int:
        return sum([len(d) for d in self.datasets.values()])
    
    def __getdataset__(self, index: int) -> tuple[int, int]:
        i = 0
        while index >= self.lengths[i]:
            index -= self.lengths[i]
            i += 1
        return i, index
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dataset_idx, index = self.__getdataset__(index)
        dataset = self.datasets[self.datasets_names[dataset_idx]]
        anchor = dataset[index]
        anchor_tensor = processor(anchor.image, return_tensors="pt", padding=True)["pixel_values"]

        positives = [transforms(anchor_tensor).squeeze() for _ in range(self.num_pos + 1)] # positves = random transforms of anchor
        anchor_tensor = positives[0]
        positives = torch.stack(positives[1:])
        # negative_indices = torch.randint(0, len(self.dataset), (self.num_neg,)).tolist()
        # negatives = [self.dataset[i] for i in negative_indices] # negatives = random graphs from dataset

        negative_space = []
        for d in self.datasets.values():
            negative_space.extend(d.graphs)
        negative_space = np.array(negative_space)

        # use numpy to sample negatives to avoid duplicates
        negatives = np.random.choice(negative_space, self.num_neg, replace=False).tolist()


        # sort negatives by distance to anchor
        negatives_2 = np.random.choice(dataset, min(self.num_neg, len(dataset)), replace=False).tolist()
        negatives_2.sort(key=lambda x: heuristic_distance(anchor, x))
        hard_negatives = negatives_2[:self.num_hard_neg]
        negatives += hard_negatives
         
        negatives_tensor = [processor(n.image, return_tensors="pt", padding=True)["pixel_values"].squeeze() for n in negatives]
        negatives = [transforms(neg).squeeze() for neg in negatives_tensor]
        negatives_tensor = torch.stack(negatives)

        return anchor_tensor, positives, negatives_tensor
    
    def get_labels(self) -> torch.Tensor:
        positives = torch.zeros(self.num_pos + 1)
        negatives = torch.arange(1, self.num_neg + self.num_hard_neg + 1)
        return torch.cat((positives, negatives))

ds = ImageTriplesDataset(datasets, config["positives"], config["negatives"], config["hard_negatives"])

import torch
from torch.nn import Linear, Sequential, ReLU, Dropout
from transformers import AutoModel, AutoImageProcessor

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        self.model = AutoModel.from_pretrained(MODEL_PATH)
        self.projection_head = Sequential(
            Linear(768, 256),
            ReLU(),
            Dropout(0.05),
            Linear(256, 32),
        )

    def forward(self, x, train = True):
        if train:
            # x = self.processor(x, return_tensors="pt", padding=True)
            x = self.model(x).pooler_output
            x = self.projection_head(x)
            return x
        else:
            # x = self.processor(x, return_tensors="pt", padding=True)
            x = self.model(x).pooler_output
            return x
        
from pytorch_metric_learning.losses import NTXentLoss
loss_func = NTXentLoss(temperature=config["temperature"])

model = Model().to(device)
if config["optimizer"] == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
elif config["optimizer"] == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])

import torch
from torch.utils.data import DataLoader
def collate_fn(batch):
    anchor, positives, negatives = batch[0]
    return torch.cat((anchor.unsqueeze(0), positives, negatives))

train_loader = DataLoader(ds, collate_fn=collate_fn, batch_size=config["batch_size"], shuffle=True, num_workers=22)

import tqdm
import logging
logging.basicConfig(level=logging.INFO, filename="train.log", filemode="w", format="%(asctime)s %(levelname)s %(message)s")

def train():
    model.train()
    total_loss = 0
    for batch_number, data in enumerate(tqdm.tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        # Get data representations
        embeddings = model(data)
        # The same index corresponds to a positive pair
        labels = ds.get_labels().to(device)
        loss = loss_func(embeddings, labels)
        loss.backward()
        total_loss += loss.item() * data.shape[0]
        optimizer.step()
    return total_loss / len(ds)


prev_loss = 20000
for epoch in range(1, config["epochs"] + 1):
    loss = train()
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    logging.info(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    wandb.log({"loss": loss})
    if loss < prev_loss:
        # save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"model_{epoch}.pt")
        prev_loss = loss
    scheduler.step()