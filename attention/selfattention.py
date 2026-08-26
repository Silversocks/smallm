import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],    # Your
    [0.55, 0.87, 0.66],     # journey
    [0.57, 0.85, 0.64],     # starts
    [0.22, 0.58, 0.33],     # with
    [0.77, 0.25, 0.10],     # one
    [0.05, 0.80, 0.55]]     # step
)

query = inputs[4]
attn_scores = torch.empty(inputs.shape[0])

for i,x_i in enumerate(inputs):
    attn_scores[i] = torch.dot(x_i,query)

# normalisation method
attn_weights = attn_scores/attn_scores.sum()

#softmax method
attn_weights = torch.exp(attn_scores)/torch.exp(attn_scores).sum(dim=0)
# or
attn_weights = torch.softmax(attn_scores,dim=0)
