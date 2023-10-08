import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch import linear
from dgl.data import KarateClubDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


class GraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, aggregator_type='mean'):
        super().__init__()
        self.sage_1 = SAGEConv(in_features, hidden_features, aggregator_type)
        self.sage_2 = SAGEConv(hidden_features, out_features, aggregator_type)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, g, x, etype=None):
        x = x.to(self.device)
        g = g.to(self.device)
        x = self.sage_1(g, x)
        x = self.sage_2(g, x)
        return F.normalize(x, p=2, dim=1)
