from glob import glob
from datasets import load_dataset

dataset = load_dataset("kblw/kialo-graphnli-images")

from torch.utils.data import DataLoader
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    # v2.RandomRotation(45),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    v2.ToTensor()
])

import torch
from torch.nn import Linear
from transformers import Swinv2Model, AutoImageProcessor

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained("../pretrained_model")
        self.model = Swinv2Model.from_pretrained("../pretrained_model")
        self.projection_head = Linear(768, 32)

    def forward(self, x, train = True):
        if train:
            augm_1 = transforms(x)
            augm_2 = transforms(x)
            x1 = self.model(augm_1).pooler_output
            x1 = self.projection_head(x1)
            x2 = self.model(augm_2).pooler_output
            x2 = self.projection_head(x2)
            return x1, x2
        else:
            # x = self.processor(x, return_tensors="pt", padding=True)
            x = self.model(**x).pooler_output
            x = self.projection_head(x)
            return x
        
from pytorch_metric_learning.losses import NTXentLoss
loss_func = NTXentLoss(temperature=0.1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

import torch
def collate_fn(batch):
    images = [torch.Tensor(example["pixel_values"]) for example in batch]
    images = torch.stack(images)
    return images

train_loader = DataLoader(dataset["train"], collate_fn=collate_fn, batch_size=32, shuffle=True, num_workers=4)

import tqdm
import logging
logging.basicConfig(filename='finetune.log',level=logging.DEBUG)

def train():
    model.train()
    total_loss = 0
    for _, data in enumerate(tqdm.tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        # Get data representations
        emb_view1, emb_view2 = model(data)
        # Prepare for loss
        embeddings = torch.cat((emb_view1, emb_view2))
        # The same index corresponds to a positive pair
        indices = torch.arange(0, emb_view1.size(0), device=emb_view2.device)
        labels = torch.cat((indices, indices))
        loss = loss_func(embeddings, labels)
        loss.backward()
        total_loss += loss.item() * data.shape[0]
        optimizer.step()
    return total_loss / len(dataset)

for epoch in range(1, 30):
    loss = train()
    logging.debug(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    scheduler.step()
    torch.save(model.state_dict(), f"../finetuned_model/epoch_{epoch}.pt")