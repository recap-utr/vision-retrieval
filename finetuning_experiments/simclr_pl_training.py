import os
import urllib.request
from copy import deepcopy
from urllib.error import HTTPError

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from datasets import load_dataset
from tqdm.notebook import tqdm
from lightning.pytorch.loggers import WandbLogger



# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/ContrastiveLearning_graphimages-twopi_testpt-noise/")
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count()

MODEL="microsoft/swinv2-tiny-patch4-window8-256"

# Setting the seed
L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

contrast_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        #transforms.RandomResizedCrop(size=256),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9)
    ]
)

from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained(MODEL)
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import torch
tt = ToTensor()
tp = ToPILImage()

def create_views(x, base_transforms, n_views):
    return torch.cat([processor(base_transforms(x.convert("RGB")), return_tensors="pt")["pixel_values"] for i in range(n_views)])

def replace_white_with_noise(image):
    # Convert image to a PyTorch tensor
    img_tensor = tt(image)

    # Create a mask for white pixels
    white_mask = (img_tensor[0] == 1) & (img_tensor[1] == 1) & (img_tensor[2] == 1)

    # Apply random noise to white pixels
    img_tensor[:, white_mask] = torch.rand_like(img_tensor[:, white_mask])

    # Convert the tensor back to PIL Image
    noisy_image = tp(img_tensor)

    return noisy_image
   
def apply_transforms(examples):
    examples["image"] = [replace_white_with_noise(x) for x in examples["image"]]
    examples["pixel_values"] = [create_views(x, contrast_transforms, 2) for x in examples["image"]]
    return examples

ds = load_dataset("kblw/graphviz_treemap")
ds.set_transform(apply_transforms)
unlabeled_data = ds["train"]
train_data_contrast = ds["test"]

from transformers import AutoModel
class SimCLR(L.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500, model=MODEL, dataset=ds):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.model = AutoModel.from_pretrained(model)
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.mlp = nn.Sequential(
            nn.Linear(768, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        # print(batch)
        imgs = batch
        # imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.model(imgs).pooler_output
        feats = self.mlp(feats)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

coalate_fn = lambda batch: torch.cat([torch.Tensor(x["pixel_values"]) for x in batch])

def train_simclr(batch_size, max_epochs=500, **kwargs):
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR"),
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
            LearningRateMonitor("epoch"),
        ],
        logger=WandbLogger(project="SimCLR_pl", log_model=True),
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SimCLR.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = SimCLR.load_from_checkpoint(pretrained_filename)
    else:
        train_loader = data.DataLoader(
            unlabeled_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=40,
            collate_fn=coalate_fn,
        )
        val_loader = data.DataLoader(
            train_data_contrast,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=40,
            collate_fn=coalate_fn,
        )
        L.seed_everything(42)  # To be reproducible
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model

simclr_model = train_simclr(
    batch_size=64, hidden_dim=64, lr=5e-4, temperature=0.07, weight_decay=1e-4, max_epochs=300
)

