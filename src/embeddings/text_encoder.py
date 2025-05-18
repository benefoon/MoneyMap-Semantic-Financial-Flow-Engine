from transformers import AutoTokenizer, AutoModel
import torch

class TextEncoder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            embeddings = self.model(**tokens).last_hidden_state[:, 0]
        return embeddings
