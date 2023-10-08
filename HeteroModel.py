import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import SAGEConv
from dgl.data import KarateClubDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from dgl.nn.pytorch import RelGraphConv
from dgl.nn.pytorch import HeteroGraphConv
from HeteroDataset import *
from HomoDataset import *


# class PinSAGE(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features, rel_num, rel_types):
#         super().__init__()
#         self.sage_1 = RelGraphConv(in_features, hidden_features, rel_num)
#         self.sage_2 = RelGraphConv(hidden_features, out_features, rel_num)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.rel_types = torch.tensor(rel_types).to(self.device)
#
#     def forward(self, g, x, etype=None):
#         x = x.to(self.device)
#         g = g.to(self.device)
#         h = self.sage_1(g, x, self.rel_types)
#         h = self.sage_2(g, h, self.rel_types)
#         w = self.sage_1.linear_r.get_weight()
#         print(w[1, :, :])
#         print(w.shape)
#         return F.normalize(h, p=2, dim=1)

class PinSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device, graph):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        conv_dict_1 = {
            rel[1]: SAGEConv(in_features, hidden_features, aggregator_type='mean') for rel in graph.canonical_etypes
        }

        conv_dict_2 = {
            rel[1]: SAGEConv(hidden_features, out_features, aggregator_type='mean') for rel in graph.canonical_etypes
        }
        self.RSAGEConv_1 = HeteroGraphConv(conv_dict_1, aggregate='mean')
        self.RSAGEConv_2 = HeteroGraphConv(conv_dict_2, aggregate='mean')
        # self.RSAGEConv_2 = HeteroGraphConv(conv_dict_2, aggregate='mean')

    def forward(self, g, x, etype=None):
        # x = x.to(self.device)
        g = g.to(self.device)
        h = self.RSAGEConv_1(g, x)
        h = self.RSAGEConv_2(g, h)
        user = h['user']
        item = h['item']
        return F.normalize(torch.cat((item, user[len(item):])), p=2, dim=1)
