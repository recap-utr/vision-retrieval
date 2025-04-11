import os
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from torchvision import transforms
from datasets import load_dataset
import wandb
from transformers import AutoImageProcessor, AutoModel
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path


PROJECT = "VisionRetrievalFineTuning"


class SimCLR(L.LightningModule):
    def __init__(
        self,
        hidden_dim,
        lr: float,
        temperature: float,
        weight_decay: float,
        basemodel: str,
        checkpoint: str,
        dataset: str,
        latent_dim: int,
        max_epochs=500,
        pretrained_model=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_model"])
        assert temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        if pretrained_model is None:
            raise ValueError("Pretrained model must be provided!")
        else:
            self.model = pretrained_model
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
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
        self.log(mode + "_loss", nll, sync_dist=True)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [
                cos_sim[pos_mask][:, None],
                cos_sim.masked_fill(pos_mask, -9e15),
            ],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), sync_dist=True)
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean(), sync_dist=True)
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean(), sync_dist=True)

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")


def main(
    base_model: str,
    batch_size: int,
    latent_dim: int,
    max_epochs: int,
    save_path: Path,
    pretrained_checkpoint: str,
    dataset_name: str,
    wandb_project: str = "",
    num_workers: int = 30,
):
    processor = AutoImageProcessor.from_pretrained(base_model)
    vis = dataset_name.split("/")[-1]

    L.seed_everything(42)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    print("Number of workers:", num_workers)

    contrast_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=256, scale=(0.4, 0.9)),
            transforms.GaussianBlur(kernel_size=9),
            transforms.RandomVerticalFlip(),
        ]
    )

    def create_views(x, base_transforms, n_views, dropout) -> torch.Tensor:
        return torch.cat(
            [
                dropout(
                    processor(base_transforms(x.convert("RGB")), return_tensors="pt")[
                        "pixel_values"
                    ]
                )
                for i in range(n_views)
            ]
        )

    dropout = nn.Dropout(p=0.1)

    def apply_transforms(examples):
        examples["pixel_values"] = [
            create_views(x, contrast_transforms, 2, dropout) for x in examples["image"]
        ]
        return examples

    ds = load_dataset(dataset_name)
    ds.set_transform(apply_transforms)
    # generate test split if not present
    if "test" not in ds:
        ds = ds["train"].train_test_split(test_size=0.1)
    unlabeled_data = ds["train"]
    train_data_contrast = ds["test"]

    coalate_fn = lambda batch: torch.cat(
        [torch.Tensor(x["pixel_values"]) for x in batch]
    )

    if wandb_project != "":
        wandb_logger = WandbLogger(project=wandb_project, log_model=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{save_path}/checkpoints", save_top_k=2, monitor="val_loss"
    )
    trainer = L.Trainer(
        default_root_dir=f"{save_path}/logs",
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_epochs=max_epochs,
        # precision="16-mixed",
        callbacks=[
            checkpoint_callback,
            # GenerateCallback(get_train_images(8), every_n_epochs=10),
            LearningRateMonitor("epoch"),
            # EarlyStopping(
            #     monitor="val_loss",
            #     patience=3,
            #     mode="min",
            #     min_delta=0.2,
            #     verbose=True,
            # ),
        ],
    )
    if wandb_project != "":
        trainer.logger = wandb_logger

    train_loader = data.DataLoader(
        unlabeled_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=coalate_fn,
    )
    val_loader = data.DataLoader(
        train_data_contrast,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=coalate_fn,
    )
    pretrained_model = AutoModel.from_pretrained(pretrained_checkpoint)
    pretrained_model.train()
    model = SimCLR(
        max_epochs=max_epochs,
        pretrained_model=pretrained_model,
        latent_dim=latent_dim,
        hidden_dim=64,
        lr=5e-4,
        temperature=0.07,
        weight_decay=1e-4,
        basemodel=base_model,
        checkpoint=pretrained_checkpoint,
        dataset=dataset_name,
    )
    trainer.fit(model, train_loader, val_loader)
    model.model.save_pretrained(f"{save_path}/model")
    wandb.finish()
    return model


if __name__ == "__main__":
    pass
