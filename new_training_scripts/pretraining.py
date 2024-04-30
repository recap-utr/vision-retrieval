import os
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel
import lightning as L
import matplotlib
import matplotlib_inline.backend_inline
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb

MODEL = "microsoft/swinv2-tiny-patch4-window8-256"
EPOCHS = 100

matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()
sns.set()

# Tensorboard extension (for visualization purposes later)
def main(dataset_name, revision=None):
    wandb_logger = WandbLogger(project="PretrainAllNew", log_model=True)
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/PretrainAllNew")
    BATCH_SIZE = 64
    LATENT_DIM = 768
    # Setting the seed
    L.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    # Transformations applied on each image => only make them a tensor
    processor = AutoImageProcessor.from_pretrained(MODEL)
    def apply_transforms(examples):
        examples["pixel_values"] = [process(image.convert("RGB")) for image in examples["image"]]
        return examples
    process = lambda x: processor(x, return_tensors="pt", normalize=True)["pixel_values"].squeeze()
    if revision == None:
        ds = load_dataset(dataset_name)
    else:
        ds = load_dataset(dataset_name, revision=revision)
    ds.set_transform(apply_transforms)
    # generate test split if not present
    if "test" not in ds:
        ds = ds["train"].train_test_split(test_size=0.1)
    train_set = ds["train"]
    test_set = ds["test"]
    col = lambda batch: torch.stack([example["pixel_values"] for example in batch])
    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, drop_last=True, pin_memory=True, num_workers=30, collate_fn=col)
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, drop_last=False, num_workers=30, collate_fn=col)
    def train_cifar(checkpoint_path, latent_dim):
        # Create a PyTorch Lightning trainer with the generation callback
        trainer = L.Trainer(
            default_root_dir=os.path.join(checkpoint_path, "cifar10_%i" % latent_dim),
            accelerator="auto",
            devices=1,
            max_epochs=EPOCHS,
            callbacks=[
                ModelCheckpoint(save_weights_only=True),
                #GenerateCallback(get_train_images(8), every_n_epochs=10),
                LearningRateMonitor("epoch"),
            ],
            logger=wandb_logger,
        )
        trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        model = Autoencoder(base_channel_size=256, latent_dim=latent_dim, model=MODEL, dataset=dataset_name, batch_size=BATCH_SIZE)
        trainer.fit(model, train_loader, test_loader)
        # Test best model on validation and test set
        test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
        result = {"test": test_result}
        return model, result

    model, _ = train_cifar(CHECKPOINT_PATH, LATENT_DIM)    
    wandb.finish()

class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
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
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
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
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)

if __name__ == "__main__":
    main("kblw/graphviz_treemap")