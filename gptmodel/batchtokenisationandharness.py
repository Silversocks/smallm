import torch
import torch.nn as nn
import tiktoken

encoder = tiktoken.get_encoding("gpt2")

batch = []
txt1 = "Hello world I am Person"
txt2 = "This is a test sentence"
print(txt1)
print(txt2)
batch.append(torch.tensor(encoder.encode(txt1)))
batch.append(torch.tensor(encoder.encode(txt2)))
batch = torch.stack(batch,dim=0)

print(batch)
