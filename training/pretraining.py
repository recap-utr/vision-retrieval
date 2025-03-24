from datasets import load_dataset, IterableDataset
from transformers import AutoImageProcessor, AutoModel
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
from lightning.pytorch.loggers import WandbLogger
import wandb



def main(dataset_name, basemodel, latent_dim, batch_size, epochs, save_path, wandb_project="", num_workers=30):
    if wandb_project != "":
        wandb.init(project=wandb_project)
        wandb_logger = WandbLogger(project=wandb_project, log_model=True)
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    # Path to the folder where the pretrained models are saved
    # Setting the seed
    L.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    processor = AutoImageProcessor.from_pretrained(basemodel)

    def apply_transforms(examples):
        examples["pixel_values"] = [
            process(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    process = lambda x: processor(x, return_tensors="pt")["pixel_values"].squeeze()
    ds = load_dataset(dataset_name)
    if isinstance(ds, IterableDataset):
        raise ValueError("Only non-iterable datasets are supported")
    ds.set_transform(apply_transforms)

    # generate test split if not present
    if "test" not in ds:
        ds = ds["train"].train_test_split(test_size=0.1)
    train_set = ds["train"]
    test_set = ds["test"]
    col = lambda batch: torch.stack([example["pixel_values"] for example in batch])
    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=col,
    )
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, drop_last=False, num_workers=30, collate_fn=col
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{save_path}/checkpoints", save_top_k=2, monitor="val_loss"
    )
    trainer = L.Trainer(
        default_root_dir=f"{save_path}/logs",
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_epochs=epochs,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min",
                min_delta=0.5,
                verbose=True,
            ),
        ],
    )
    if wandb_project != "":
        trainer.logger = wandb_logger

    model = Autoencoder(
        base_channel_size=256,
        latent_dim=latent_dim,
        model=basemodel,
        dataset=dataset_name,
        batch_size=batch_size,
    )
    trainer.fit(model, train_loader, test_loader)
    # Test best model on validation and test set
    trainer.test(model, dataloaders=test_loader, verbose=False)
    model.encoder.save_pretrained(save_path)
    wandb.finish()
    return model


class Encoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        base_channel_size: int,
        latent_dim: int,
        act_fn: torch.nn.Module = nn.GELU(),
    ):
        """Encoder.

        Args:
            num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            latent_dim : Dimensionality of latent representation z
            act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(
                num_input_channels, c_hid, kernel_size=3, padding=1, stride=2
            ),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(
                c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2
            ),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2
            ),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, img_size: int, latent_dim: int):
        """Decoder.

        Args:
            num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            img_size : The height/width of the input images
            latent_dim : Dimensionality of latent representation z
        """
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_input_channels * img_size * img_size)
        self.img_size = img_size

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, self.img_size, self.img_size)
        return x


class Autoencoder(L.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        dataset: str,
        model: str,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 256,
        height: int = 256,
        batch_size: int = 64,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = AutoModel.from_pretrained(model)
        self.decoder = decoder_class(num_input_channels, 32, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x).pooler_output
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)."""
        x = batch  # We do not need the labels
        resized_x = F.interpolate(x, size=(32, 32))
        x_hat = self.forward(x)
        loss = F.mse_loss(resized_x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
        return loss


if __name__ == "__main__":
    pass
