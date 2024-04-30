from sentence_transformers import SentenceTransformer
import torch
model = SentenceTransformer('all-mpnet-base-v2')

def embedd_texts(texts: list[str]) -> list[torch.Tensor]:
    embeddings = model.encode(texts)
    return embeddings