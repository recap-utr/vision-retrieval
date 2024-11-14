VIS = "space reclaiming icicle chart"
IMG_PLACEHOLDER = "data:image/png;base64,"
NUM_SAMPPLES = 500
IMAGES_PER_SAMPLE = 2

import base64
import jsonlines
import random
import os
from PIL import Image
from torchvision import transforms
import io
from tqdm import tqdm
import shutil


def image_to_base64(image) -> str:
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

def create_contrastive_views(image_path):
    original_image = Image.open(image_path).convert('RGB')
    transform = contrast_transforms
    
    contrastive_views = []
    for _ in range(IMAGES_PER_SAMPLE):
        contrastive_views.append(transform(original_image))
    
    return contrastive_views
                                                            


def build_sample(images: list, same: bool):
    d = {
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that identifies visualizations of argument graphs.",
            },
            {
                "role": "user",
                "content": f"Take a look at the following images in {VIS} visualization. Are they visualizations of the same argument graph or do they describe different graphs? Answer only with 'same' or 'different'.",
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


if __name__ == "__main__":
    folder_path = "data/arg_finetune_logical_srip/srip2/train"
    output_path = "data/oai-finetuning/"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # Get a list of all image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith(".png")]
    print(f"Found {len(image_files)} images")
    random.shuffle(image_files)

    # Randomly select two images for each sample ("different" samples)
    samples = []
    os.makedirs(output_path + "different", exist_ok=True)
    for i in tqdm(range(NUM_SAMPPLES)):
        if len(image_files) < IMAGES_PER_SAMPLE:
            raise ValueError("not enough samples left")
        random_images = [f"{folder_path}/{image_files.pop()}" for _ in range(IMAGES_PER_SAMPLE)]
        sample_dir = f"{output_path}different/{i}"
        os.makedirs(sample_dir, exist_ok=True)
        [shutil.copy2(image, sample_dir) for image in random_images]
        random_images = [Image.open(f).convert("RGB") for f in random_images]
        
        samples.append(build_sample(random_images, same=False))
        
    print("Different samples done")
    print(f"Images left: {len(image_files)}")
    # "same" samples
    os.makedirs(output_path + "same", exist_ok=True)
    for i in tqdm(range(NUM_SAMPPLES)):
        random_image = f"{folder_path}/{image_files.pop()}"
        sample_dir = f"{output_path}same/{i}"
        os.makedirs(sample_dir, exist_ok=True)

        images = create_contrastive_views(random_image)
        [image.save(f"{sample_dir}/{i}.png") for i, image in enumerate(images)]

        samples.append(
            build_sample(create_contrastive_views(random_image), same=True)
        )
    print("Same samples done")
    # Write the samples to a JSONL file
    up_to = len(samples) // 10
    with jsonlines.open(output_path + "contrastive_test.jsonl", "w") as writer:
        for sample in samples[:up_to]:
            writer.write(sample)
    print("Done writing samples to contrastive_test.jsonl")
    with jsonlines.open(output_path + "contrastive_train.jsonl", "w") as writer:
        for sample in samples[up_to:]:
            writer.write(sample)
    print("Done writing samples to contrastive_train.jsonl")
