import torch
import torch.nn as nn

class selfattention(nn.Module): # simple self attention
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in,d_out))
        self.W_key = nn.Parameter(torch.rand(d_in,d_out))
        self.W_value = nn.Parameter(torch.rand(d_in,d_out))

    def forward(self,x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
        context_vec = attn_weights @ values
        return context_vec

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

class MultiHeadAttention(nn.Module): # use this class for anything practical
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        self.d_out=d_out
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)       
        self.dropout=nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out,d_out) # to combine attention heads outputs
        self.head_dim=self.d_out//num_heads
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        b,num_tokens,d = inputs.shape
        query =  self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values = values.view(b,num_tokens,self.num_heads,self.head_dim)
        query = query.view(b,num_tokens,self.num_heads,self.head_dim)
        keys = keys.transpose(1,2)
        query = query.transpose(1,2)
        values = values.transpose(1,2)
        attn_scores = query @ keys.transpose(2,3)
        mask = self.mask.bool()[:num_tokens,:num_tokens]
        attn_scores.masked_fill(mask,-torch.inf)
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
        b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)
        return context_vec
    # quick notes: the multi head attention sample class does the heaviest computation over iterating through a list, making it considerably slower and more inefficient. The above class effectively does the same thing using just 1 large matrix split into smaller matrices such that a single matrix multiplication is enough for the entire thing

if __name__ == "__main__": #testing purposes
    inputs=torch.rand(1000,6)
    batch= torch.stack((inputs,inputs),dim=0)
    torch.manual_seed(123)
    context_length = 1000
    d_in = 6
    d_out = 3
    ca = CausalAttention(d_in, d_out, context_length, 0.2)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape) 
