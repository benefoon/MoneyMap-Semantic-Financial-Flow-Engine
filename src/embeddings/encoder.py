from typing import List
import torch
import torch.nn as nn
from .text_encoder import TextEncoder
from .numeric_encoder import NumericEncoder
from .fusion import fuse_embeddings

class TransactionEncoder(nn.Module):
    def __init__(self, text_model='sentence-transformers/all-MiniLM-L6-v2', numeric_dim=10, embedding_dim=384):
        super(TransactionEncoder, self).__init__()
        self.text_encoder = TextEncoder(model_name=text_model)
        self.numeric_encoder = NumericEncoder(numeric_dim, embedding_dim)

    def forward(self, text_inputs: List[str], numeric_inputs: torch.Tensor):
        text_emb = self.text_encoder.encode(text_inputs)
        num_emb = self.numeric_encoder(numeric_inputs)
        return fuse_embeddings(text_emb, num_emb)
