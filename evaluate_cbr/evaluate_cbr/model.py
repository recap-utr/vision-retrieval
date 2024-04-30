from typing import Any
from transformers import AutoModel, AutoImageProcessor
import torch
from PIL import Image

class Model:
    def __init__(self, modelname: str) -> None:
        self.model: AutoModel = AutoModel.from_pretrained(modelname)
        try:
            self.processor = AutoImageProcessor.from_pretrained(modelname)
        except:
            self.processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    def embedd(self, image: torch.Tensor):
        # processed = processor(image, return_tensors="pt")
        return self.model(image).pooler_output.squeeze()
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.embedd(*args, **kwds)
