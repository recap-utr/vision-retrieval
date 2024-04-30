import os
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from datasets import load_dataset
import wandb
from new_training_scripts.pretraining import Autoencoder
from transformers import AutoImageProcessor


BASEMODEL = "microsoft/swinv2-tiny-patch4-window8-256"
processor = AutoImageProcessor.from_pretrained(BASEMODEL)
EPOCHS = 100


class SimCLR(L.LightningModule):
    def __init__(
        self,
        hidden_dim,
        lr,
        temperature,
        weight_decay,
        basemodel: str,
        checkpoint: str,
        dataset: str,
        revision: str,
        max_epochs=500,
        pretrained_model=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert (
            self.hparams.temperature > 0.0
        ), "The temperature must be a positive float!"
        # Base model f(.)
        if pretrained_model is None:
            from transformers import AutoModel, AutoImageProcessor

            ae = Autoencoder.load_from_checkpoint(checkpoint)
            processor = AutoImageProcessor.from_pretrained(BASEMODEL)
            self.model = ae.encoder
        else:
            self.model = pretrained_model
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.mlp = nn.Sequential(
            nn.Linear(768, 4 * hidden_dim),
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
        self.log(mode + "_loss", nll)
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
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")


def main(checkpoint_path, dataset_name, revision=None):

    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = os.environ.get(
        "PATH_CHECKPOINT", "saved_models/SimCLR_Treemap_SAT/"
    )
    # In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
    # workers as possible in a data loader, which corresponds to the number of CPU cores
    NUM_WORKERS = os.cpu_count()

    BASEMODEL = "microsoft/swinv2-tiny-patch4-window8-256"
    BATCH_SIZE = 64

    # Setting the seed
    L.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("Device:", device)
    print("Number of workers:", NUM_WORKERS)

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

    if revision == None:
        ds = load_dataset(dataset_name)
    else:
        ds = load_dataset(dataset_name, revision=revision)
    ds.set_transform(apply_transforms)
    # generate test split if not present
    if "test" not in ds:
        ds = ds["train"].train_test_split(test_size=0.1)
    unlabeled_data = ds["train"]
    train_data_contrast = ds["test"]

    coalate_fn = lambda batch: torch.cat(
        [torch.Tensor(x["pixel_values"]) for x in batch]
    )

    from lightning.pytorch.loggers import WandbLogger

    wandb_logger = WandbLogger(project="FinetuneAllNew", log_model=True)

    def train_simclr(batch_size, max_epochs=500, **kwargs):
        trainer = L.Trainer(
            default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR"),
            accelerator="auto",
            devices=1,
            max_epochs=max_epochs,
            callbacks=[
                ModelCheckpoint(
                    save_weights_only=True, mode="max", monitor="val_acc_top5"
                ),
                LearningRateMonitor("epoch"),
            ],
            logger=wandb_logger,
            log_every_n_steps=16,
        )
        trainer.logger._default_hp_metric = (
            None  # Optional logging argument that we don't need
        )

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
                num_workers=30,
                collate_fn=coalate_fn,
            )
            val_loader = data.DataLoader(
                train_data_contrast,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=30,
                collate_fn=coalate_fn,
            )
            L.seed_everything(42)  # To be reproducible
            model = SimCLR(max_epochs=max_epochs, **kwargs)
            trainer.fit(model, train_loader, val_loader)
            # Load best checkpoint after training
            model = SimCLR.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )

        return model

    simclr_model = train_simclr(
        batch_size=64,
        hidden_dim=64,
        lr=5e-4,
        temperature=0.07,
        weight_decay=1e-4,
        max_epochs=EPOCHS,
        basemodel=BASEMODEL,
        checkpoint=checkpoint_path,
        dataset=dataset_name,
        revision=revision,
    )
    wandb.finish()


if __name__ == "__main__":
    main("../saved_models/PretrainAllNew/treemap_sat.ckpt", "kblw/graphviz_treemap")
