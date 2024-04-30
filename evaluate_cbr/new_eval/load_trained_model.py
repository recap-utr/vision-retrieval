import torch
from transformers import AutoModel, AutoImageProcessor

def load_pt(basemodel, version, statedict_file="./models/pt_statedicts.pt"):
    model = AutoModel.from_pretrained(basemodel)
    d = torch.load(statedict_file, map_location=torch.device('cpu'))
    model.load_state_dict(d[version])
    # processor = AutoImageProcessor.from_pretrained(basemodel)
    return model #, processor
def load_ft(basemodel, version, statedict_file="./models/ft_statedicts.pt"):
    model = AutoModel.from_pretrained(basemodel)
    d = torch.load(statedict_file, map_location=torch.device('cpu'))
    model.load_state_dict(d[version])
    # processor = AutoImageProcessor.from_pretrained(basemodel)
    return model #, processor
