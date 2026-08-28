import torch

inputs=torch.rand(1000,3)

attn_scores = torch.empty(inputs.shape[0],inputs.shape[0])

for j,query in enumerate(inputs):
    for i,x_i in enumerate(inputs):
        attn_scores[j,i]=torch.dot(query,x_i)

attn_weights = attn_scores/attn_scores.sum(dim=1) 
print(str(attn_weights)+", sum="+str(attn_scores.sum()))
