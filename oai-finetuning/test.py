import os
from PIL import Image
from glob import glob

def check_image_dimensions(directory):
    dimensions = None
    for filename in glob(f"{directory}/**/*.png", recursive=True):
        with Image.open(filename) as img:
            if dimensions is None:
                dimensions = img.size
            elif img.size != dimensions:
                return False, dimensions
    return True, dimensions

directory = 'data/oai-finetuning'
same, dimensions = check_image_dimensions(directory)
if same:
    print(f"All images have the same dimensions. Dimensions: {dimensions}")
else:
    print("Not all images have the same dimensions.")