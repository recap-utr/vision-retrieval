import os
from urllib.error import HTTPError

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm.notebook import tqdm
from new_training_scripts.pretraining import Autoencoder
from transformers import AutoImageProcessor
from tqdm import tqdm

from datasets import load_dataset

matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()
sns.set()

MODEL = "microsoft/swinv2-tiny-patch4-window8-256"


def vis_pretrain(checkpoint, dataset_name, revision=None):
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)

    # Setting the seed
    L.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("Device:", device)

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    def apply_transforms(examples):
        examples["pixel_values"] = [
            transform(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    # Loading the training dataset. We need to split it into a training and validation part
    L.seed_everything(42)
    processor = AutoImageProcessor.from_pretrained(MODEL)

    def apply_transforms(examples):
        examples["pixel_values"] = [
            process(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    process = lambda x: processor(x, return_tensors="pt", normalize=True)[
        "pixel_values"
    ].squeeze()
    ds = load_dataset(dataset_name, revision=revision)
    ds.set_transform(apply_transforms)
    train_set = ds["train"]
    # We define a set of data loaders that we can use for various purposes later.
    # train_loader = data.DataLoader(train_set, batch_size=128, drop_last=True, pin_memory=True, num_workers=20, collate_fn=col)
    # test_loader = data.DataLoader(test_set, batch_size=128, drop_last=False, num_workers=20, collate_fn=col)
    path = f"training_visualizations/pretrain/{dataset_name.replace('/', '_')}"
    if revision is not None:
        path += f"_{revision}"
    os.makedirs(path, exist_ok=True)

    def visualize_reconstructions(model, input_imgs, image_name):
        # Reconstruct images
        model.eval()
        with torch.no_grad():
            reconst_imgs = model(input_imgs.to(model.device))
        reconst_imgs = reconst_imgs.cpu()
        reconst_imgs = F.interpolate(reconst_imgs, size=(256, 256))
        # Plotting
        imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
        grid = torchvision.utils.make_grid(
            imgs, nrow=4, normalize=True, value_range=(-1, 1)
        )
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(7, 4.5))
        plt.title("Reconstructions produced by the vision model")
        plt.imshow(grid)
        plt.axis("off")
        plt.savefig(f"{path}/{image_name}.png")
        plt.close()

    model = Autoencoder.load_from_checkpoint(checkpoint)

    def get_train_images(amt):
        res = []
        for d in train_set:
            res.append(d["pixel_values"])
            if len(res) == amt:
                break
        return torch.stack(res)

    visualize_reconstructions(model, get_train_images(4), "reconstr")
    rand_imgs = torch.rand(2, 3, 256, 256) * 2 - 1
    visualize_reconstructions(model, rand_imgs, "random_reconstr")
    plain_imgs = torch.zeros(4, 3, 256, 256)

    # Single color channel
    plain_imgs[1, 0] = 1
    # Checkboard pattern
    plain_imgs[2, :, :16, :16] = 1
    plain_imgs[2, :, 16:, 16:] = -1
    # Color progression
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256))
    plain_imgs[3, 0, :, :] = xx
    plain_imgs[3, 1, :, :] = yy

    visualize_reconstructions(model, plain_imgs, "patterns_reconstr")


def vis_finetune(checkpoint, dataset_name, revision=None):
    L.seed_everything(42)
    contrast_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=256, scale=(0.4, 0.9)),
            transforms.GaussianBlur(kernel_size=9),
            transforms.RandomVerticalFlip(),
        ]
    )
    processor = AutoImageProcessor.from_pretrained(MODEL)

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
    unlabeled_data = ds["train"]
    path = f"training_visualizations/finetune/{dataset_name.replace('/', '_')}"
    if revision is not None:
        path += f"_{revision}"
    os.makedirs(path, exist_ok=True)

    def visualize_training_sample(image_name):
        NUM_IMAGES = 10
        imgs = torch.cat(
            [
                torch.Tensor(unlabeled_data[idx]["pixel_values"])
                for idx in range(NUM_IMAGES)
            ]
        )
        print(imgs.shape)
        img_grid = torchvision.utils.make_grid(
            imgs, nrow=NUM_IMAGES, normalize=True, pad_value=0.9
        )
        img_grid = img_grid.permute(1, 2, 0)

        plt.figure(figsize=(10, 5))
        plt.title("Samples from the fine-tuning data set")
        plt.imshow(img_grid)
        plt.axis("off")
        plt.savefig(f"{path}/{image_name}.png")
        plt.close()

    visualize_training_sample("training_samples")


if __name__ == "__main__":
    pretrain_runs = [
        (
            "../saved_models/PretrainAllNew/dot_layout.ckpt",
            "kblw/pretraining_samples_large",
        ),
        ("../saved_models/PretrainAllNew/twopi.ckpt", "kblw/graphimages_twopi"),
        (
            "../saved_models/PretrainAllNew/treemap_weak.ckpt",
            "kblw/graphviz_treemap",
            "278679f",
        ),
        ("../saved_models/PretrainAllNew/treemap_sat.ckpt", "kblw/graphviz_treemap"),
    ]

    for run in tqdm(pretrain_runs):
        vis_pretrain(*run)
    print("Pretraining images done.")
    finetune_runs = [
        ("../saved_models/FinetuneAllNew/dot_layout_simclr.ckpt", "kblw/ft-images"),
        ("../saved_models/FinetuneAllNew/twopi_simclr.ckpt", "kblw/graphimages_twopi"),
        (
            "../saved_models/FinetuneAllNew/treemap_weak_simclr.ckpt",
            "kblw/graphviz_treemap",
            "278679f",
        ),
        (
            "../saved_models/FinetuneAllNew/treemap_sat_simclr.ckpt",
            "kblw/graphviz_treemap",
        ),
    ]

    for run in tqdm(finetune_runs):
        vis_finetune(*run)
