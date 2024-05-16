from glob import glob
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool


def resize_batch(batch):
    resized_images = []
    for img in batch:
        try:
            im = Image.open(img).convert("RGB")
            im = im.resize((256, 256))
            im.save(img)
            im.close()
            resized_images.append(img)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Could not resize {img}")
    return resized_images


if __name__ == "__main__":
    images = glob("../data/random_logical_srip/**/*.png", recursive=True)
    batch_size = 4000  # Set the desired batch size
    batches = [images[i : i + batch_size] for i in range(0, len(images), batch_size)]

    with Pool() as pool:
        resized_images = list(tqdm(pool.map(resize_batch, batches), total=len(batches)))
