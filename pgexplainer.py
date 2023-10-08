from dataset import *
from model import *
import argparse
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import tqdm
import time
import torch
import collections
import numpy as np
import torch.nn as nn
import networkx as nx
from math import sqrt
from torch import Tensor
from textwrap import wrap
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Tuple, List, Dict, Optional
from torch_geometric.datasets import MoleculeNet
from rdkit import Chem


class PGExplainer(nn.Module):
    def __init__(self, model, in_features, hidden_features, device, graph):
        super().__init__()
        self.explainers = {}
        for etype in graph.canonical_etypes:
            layer = nn.Sequential(
                nn.Linear(in_features=2 * in_features, out_features=hidden_features),  # 第一个线性层
                nn.ReLU(),  # 非线性激活函数
                nn.Linear(in_features=hidden_features, out_features=1)  # 第二个线性层
            )
            self.explainers[etype](layer)
        self.node_dict = {node_type: graph.nodes[node_type].data['feat'].to(device).to(torch.float32) for node_type in
                          graph.ntypes}
        self.embed = model(graph, self.node_dict)
        self.graph = graph
        self.hops = 2
        self.top_k = 3

    def get_k_hop_subgraph(self, node, k, etype):
        res = set()
        if k != 0:
            in_nodes = self.graph.in_nodes(node, etype=etype)
            for in_node in list(in_nodes):
                tmp = self.get_k_hop_subgraph(in_node, k - 1, etype)
                res = res | set(tmp)
        return res

    def forward(self, head, tail):
        head_rel_dict = collections.defaultdict(list)
        tail_rel_dict = collections.defaultdict(list)
        head_top_k = collections.defaultdict(list)
        tail_top_k = collections.defaultdict(list)
        for etype in self.graph.canonical_etypes:
            in_subgraph_head = self.get_k_hop_subgraph(head, self.hops, etype)
            in_subgraph_tail = self.get_k_hop_subgraph(tail, self.hops, etype)

            # get in subgraph of head node
            for node in in_subgraph_head:
                edge_emb = torch.cat((self.embed[head], self.embed[node])).to(self.device)
                edge_emb_weight = self.explainers[etype](edge_emb).to('cpu')
                head_rel_dict[etype].append((node, edge_emb_weight))
            head_sorted_tuples = sorted(head_rel_dict[etype], key=lambda x: x[1], reverse=True)
            head_top_k[etype[0]] = [item[0] for item in head_sorted_tuples[:self.top_k]]

            # do the same for tail node
            for node in in_subgraph_tail:
                edge_emb = torch.cat((self.embed[tail], self.embed[node])).to(self.device)
                edge_emb_weight = self.explainers[etype](edge_emb).to('cpu')
                tail_rel_dict[etype].append((node, edge_emb_weight))
            tail_sorted_tuples = sorted(head_rel_dict[etype], key=lambda x: x[1], reverse=True)
            tail_top_k[etype[0]] = [item[0] for item in tail_sorted_tuples[:self.top_k]]

        mask_node_dict = copy.deepcopy(self.node_dict)
        for key in head_top_k:
            mask = torch.tensor([i in head_top_k[key] for i in range(mask_node_dict[key].size(0))])
            mask_node_dict[key][mask] = 0

        new_embed = self.model(self.graph, mask_node_dict)
        head_emb = new_embed[head]
        tail_emb = new_embed[tail]

        return head_emb, tail_emb
