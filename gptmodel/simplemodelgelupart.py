import torch.nn as nn
import ../activations/gelu.py

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # a larger dimension is used in hiddenlayer because it expands the representation space, allowing for potentially more meaningful exploration for the model
        GELU(),
        nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), # The reason the layer has same input and output dimensions is because it becomes easier to combine with other layers without changing intermittent dimensions.
        )
    
    def forward(self, x):
        return self.layers(x)
