from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch import HeteroGraphConv
from dataset import *


class PinSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device, graph):
        super().__init__()
        self.device = device
        conv_dict_1 = {
            rel[1]: SAGEConv(in_features, hidden_features, aggregator_type='mean') for rel in graph.canonical_etypes
        }

        conv_dict_2 = {
            rel[1]: SAGEConv(hidden_features, out_features, aggregator_type='mean') for rel in graph.canonical_etypes
        }
        self.RSAGEConv_1 = HeteroGraphConv(conv_dict_1, aggregate='mean')
        self.RSAGEConv_2 = HeteroGraphConv(conv_dict_2, aggregate='mean')

    def forward(self, g, x, etype=None):
        g = g.to(self.device)
        h = self.RSAGEConv_1(g, x)
        h = self.RSAGEConv_2(g, h)
        return h
