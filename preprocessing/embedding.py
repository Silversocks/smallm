import torch

vocab_size=50257
output_dim=256

embedding = torch.nn.Embedding(vocab_size,output_dim)
