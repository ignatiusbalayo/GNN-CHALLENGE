from dgl.nn import GATConv
import torch.nn as nn

class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super().__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads)
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, h):
        h = self.layer1(g, h).flatten(1)
        h = self.layer2(g, h).mean(1)
        return h