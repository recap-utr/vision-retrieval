from new_training_scripts.pretraining import Autoencoder
from new_training_scripts.finetuning import SimCLR
import torch
from tqdm import tqdm

pt_models = [
    "../saved_models/PretrainAllNew/new/dot_layout_v1.ckpt",
    "../saved_models/PretrainAllNew/new/twopi_layout_v2.ckpt",
    "../saved_models/PretrainAllNew/new/treemap_weak_v3.ckpt",
    "../saved_models/PretrainAllNew/new/treemap_sat_v4.ckpt",
]
ft_models = [
    "../saved_models/FinetuneAllNew/ft_dot_v1.ckpt",
    "../saved_models/FinetuneAllNew/ft_twopi_v2.ckpt",
    "../saved_models/FinetuneAllNew/ft_treemap_weak_v3.ckpt",
    "../saved_models/FinetuneAllNew/ft_treemap_sat_v4.ckpt",
]
counter = 1
d = {}
pl_models = []
for model in tqdm(pt_models):
    m = Autoencoder.load_from_checkpoint(model)
    pl_models.append(m.encoder)
    d[f"v{counter}"] = m.encoder.state_dict()
    counter += 1
torch.save(d, "../saved_models/PretrainAllNew/new/pt_statedicts.pt")
for idx, model in enumerate(tqdm(ft_models)):
    m = SimCLR.load_from_checkpoint(model, pretrained_model=pl_models[idx])
    d[f"v{counter}"] = m.model.state_dict()
    counter += 1
torch.save(d, "../saved_models/FinetuneAllNew/ft_statedicts.pt")
