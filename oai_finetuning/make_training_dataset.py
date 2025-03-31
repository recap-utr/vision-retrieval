import base64
import jsonlines
import random
import os
from PIL import Image
from torchvision import transforms
import io
from tqdm import tqdm
import shutil
import typer
from pathlib import Path
from typing_extensions import Annotated
from PIL import Image

IMG_PLACEHOLDER = "data:image/png;base64,"

app = typer.Typer()


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return IMG_PLACEHOLDER + base64.b64encode(buffered.getvalue()).decode("utf-8")


contrast_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=1000, scale=(0.4, 0.9)),
        transforms.GaussianBlur(kernel_size=9),
        transforms.RandomVerticalFlip(),
    ]
)


def create_contrastive_views(image_path: str, images_per_sample: int):
    original_image = Image.open(image_path).convert("RGB")
    transform = contrast_transforms

    contrastive_views = []
    for _ in range(images_per_sample):
        contrastive_views.append(transform(original_image))

    return contrastive_views


def build_sample(images: list, same: bool, visualization: str):
    d = {
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that identifies visualizations of argument graphs.",
            },
            {
                "role": "user",
                "content": f"Take a look at the following images in {visualization} visualization. Are they visualizations of the same argument graph or do they describe different graphs? Answer only with 'same' or 'different'.",
            },
        ]
    }
    for image in images:
        d["messages"].append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_base64(image),
                        },
                    }
                ],
            }
        )
    d["messages"].append(
        {"role": "assistant", "content": "same" if same else "different"}
    )
    return d


@app.command()
def create_training_dataset(
    input_folder: Path,
    output_folder: Path,
    visualization: Annotated[
        str,
        typer.Argument(
            help="The name of the visualization. This will be included in the task prompt and might help the LLM to interpret the images."
        ),
    ] = "space reclaiming icicle chart",
    num_samples: int = 500,
    images_per_sample: int = 2,
):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    # Get a list of all image files in the folder
    image_files = [file for file in os.listdir(input_folder) if file.endswith(".png")]
    if len(image_files) < num_samples * (images_per_sample + 1):
        raise ValueError("Not enough images in the input folder")
    print(f"Found {len(image_files)} images")
    random.shuffle(image_files)

    # Randomly select two images for each sample ("different" samples)
    samples = []
    for i in tqdm(range(num_samples)):
        if len(image_files) < images_per_sample:
            raise ValueError("not enough samples left")
        random_images = [
            f"{input_folder}/{image_files.pop()}" for _ in range(images_per_sample)
        ]
        sample_dir = f"{output_folder}/different/{i}"
        os.makedirs(sample_dir, exist_ok=True)
        [shutil.copy2(image, sample_dir) for image in random_images]
        random_images = [Image.open(f).convert("RGB") for f in random_images]

        samples.append(
            build_sample(random_images, same=False, visualization=visualization)
        )

    print("Different samples done")
    print(f"Images left: {len(image_files)}")
    # "same" samples
    for i in tqdm(range(num_samples)):
        random_image = f"{input_folder}/{image_files.pop()}"
        sample_dir = f"{output_folder}/same/{i}"
        os.makedirs(sample_dir, exist_ok=True)

        images = create_contrastive_views(random_image, images_per_sample)
        [image.save(f"{sample_dir}/{i}.png") for i, image in enumerate(images)]

        samples.append(
            build_sample(
                create_contrastive_views(random_image, images_per_sample),
                same=True,
                visualization=visualization,
            )
        )
    print("Same samples done")
    # Write the samples to a JSONL file
    up_to = len(samples) // 10
    with jsonlines.open(output_folder / "contrastive_test.jsonl", "w") as writer:
        for sample in samples[:up_to]:
            writer.write(sample)
    print("Done writing samples to contrastive_test.jsonl")
    with jsonlines.open(output_folder / "contrastive_train.jsonl", "w") as writer:
        for sample in samples[up_to:]:
            writer.write(sample)
    print("Done writing samples to contrastive_train.jsonl")


if __name__ == "__main__":
    app()
