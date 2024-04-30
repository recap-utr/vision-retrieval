import torch
from byol_pytorch import BYOL
from transformers import Swinv2Model, AutoImageProcessor
import wandb

BASE_PATH = "/home/kilian/ba/data"
MODEL_PATH = "microsoft/swinv2-tiny-patch4-window8-256"

model = Swinv2Model.from_pretrained(MODEL_PATH)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

config = {
    "optimizer": "Adam",
    "lr": 3e-4,
    "image_size": 256,
    "epochs": 50,
    "batch_size": 64,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

wandb.init(project="byol_finetuning", notes="BYOL finetuning", config=config)
import arguebuf as ab
import os
from PIL import Image
from evaluate_cbr.visualize import export_graph

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
    v2.Resize((config["image_size"], config["image_size"]), antialias=True),
    v2.ElasticTransform(fill=1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

from torch.utils.data import Dataset

class ImageGraphDataset(Dataset):

    def __init__(self, image_paths: list[str]) -> None:
        self.images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        self.pixel_values = [processor(image, return_tensors="pt").pixel_values for image in self.images]
        

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.pixel_values[index]
    

from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

from glob import glob
from time import time

start = time()

# remove microtexts from finetuning data

graph_paths = {"araucaria": "json", "iac": "json", 
               "kialo-graphnli": "json",  
               "persuasive-essays": "ann", "qt30": "json", "us-2016": "json"}

image_paths = [glob(f"{BASE_PATH}/{k}-images/*.png") for k in graph_paths.keys()]
# combine into one list
image_paths = [item for sublist in image_paths for item in sublist]

ds = ImageGraphDataset(image_paths)
print(f"Loading took {time() - start} seconds")

import torch
import numpy as np


from torch.utils.data import DataLoader
def collate_fn(batch):
    return torch.cat(batch, dim=0)

train_loader = DataLoader(ds, collate_fn=collate_fn, batch_size=config["batch_size"], shuffle=True, num_workers=22)

model = Swinv2Model.from_pretrained(MODEL_PATH)
# model.to(device)
import tqdm
import math
learner = BYOL(
    model,
    image_size = config["image_size"],
    hidden_layer = 'pooler',
    augment_fn=transforms,
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

last_loss = math.inf
for epoch in range(config["epochs"]):
    total_loss = 0
    for _, data in enumerate(tqdm.tqdm(train_loader)):
        data = data.to(device)
        loss = learner(data)
        # print(f'Loss: {loss}')
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        # wandb.log({"loss": loss})
        total_loss += loss.item()
    if total_loss < last_loss:
        last_loss = total_loss
        model.save_model(f"byol_{epoch}")
        torch.save({"optimizer": opt.state_dict()}, f"byol_{epoch}/byol_opt.pt")
        print("Saved model")