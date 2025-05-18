import torch.nn.functional as F

def fuse_embeddings(text_emb, num_emb):
    return F.normalize(text_emb + num_emb, p=2, dim=1)
