import torch
import torch.nn as nn
inputs=torch.rand(1000,3)

attn_scores = torch.empty(inputs.shape[0],inputs.shape[0])

for j,query in enumerate(inputs):
    for i,x_i in enumerate(inputs):
        attn_scores[j,i]=torch.dot(query,x_i)

attn_weights = attn_scores/attn_scores.sum(dim=1) 
print(str(attn_weights)+", sum="+str(attn_scores.sum()))

class CausalAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,qkv_bias=False):
        super().__init__()
        self.d_out=d_out
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)       
        self.dropout=nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        b,num_tokens,d_in=x.shape
        query =  self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = query @ keys.transpose(1,2)
        masked = attn_scores.masked_fill(self.mask[:num_tokens,:num_tokens].bool(),-torch.inf)
        attn_weights = torch.softmax(masked/keys.shape[-1]**0.5,dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

class MultiHeadAttentionNonParallel(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in,d_out,context_length,dropout,qkv_bias=qkv_bias) for i in range(num_heads)]
        )
        # total returns dim_out * num_heads dimensional embeding
