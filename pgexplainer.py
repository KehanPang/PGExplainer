import torch
import torch.nn as nn
import random


class PGExplainer(nn.Module):
    def __init__(self, model, in_features, hidden_features, device, graph, feat):
        super().__init__()
        self.graph = graph
        self.model = model
        self.model.eval()

        # for etype in self.graph.canonical_etypes:
        #     layer = nn.Sequential(
        #         nn.Linear(in_features=2 * in_features, out_features=hidden_features),  # 第一个线性层
        #         nn.ReLU(),  # 非线性激活函数
        #         nn.Linear(in_features=hidden_features, out_features=1),  # 第二个线性层
        #     )
        #     self.explainers[etype] = layer.to(device)
        self.explainers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),  # 第一个线性层
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),  # 第二个线性层
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_features, out_features=1, bias=True),  # 第三个线性层
        ).to(device)
        self.k_neighbours = 500
        self.hops = 3
        self.feat = feat
        self.device = device
        self.sub_graph_num = 10

    def get_masked_emb(self, mask):
        node_dict = {}
        # mask_vec = mask_vec.unsqueeze(-1)
        for node_type in self.graph.ntypes:
            feat = self.graph.nodes[node_type].data['feat'].to(self.device).to(torch.float32)
            # mask_feat = mask_vec[:len(feat)] * feat
            node_dict[node_type] = mask[:len(feat)] * feat
        return node_dict

    def get_top_k_nodes_mask(self, node):
        in_subgraph = {}

        def get_hops_neighbor(vertex, hops):
            if hops <= 0:
                return
            else:
                in_nodes = []
                for etype in self.graph.canonical_etypes:
                    try:
                        etype_nodes = list(self.graph.predecessors(vertex, etype=etype))
                        in_nodes += etype_nodes
                    except:
                        continue
                    if len(in_nodes) != 0:
                        break

                sample = random.sample(in_nodes, min(len(in_nodes), self.k_neighbours))
                for i in sample:
                    in_subgraph[i] = 1

                for vertex in sample:
                    get_hops_neighbor(vertex, hops - 1)
                return in_nodes

        get_hops_neighbor(node, self.hops)

        return torch.tensor(list(in_subgraph.keys())).to(self.device)

    def forward(self, head, tail):
        head_edge_emb = self.feat[head] * self.feat
        tail_edge_emb = self.feat[tail] * self.feat

        head_mask = self.explainers(head_edge_emb)
        tail_mask = self.explainers(tail_edge_emb)

        # head_top_k = self.get_top_k_nodes_mask(head)
        # tail_top_k = self.get_top_k_nodes_mask(tail)
        # mask_on_vec = torch.zeros(len(self.feat), dtype=torch.float32).to(self.device)
        #
        # mask_on_vec.scatter_(0, head_top_k, 1)
        # mask_on_vec.scatter_(0, tail_top_k, 1)
        #
        # mask_off_vec = torch.ones(len(self.feat), dtype=torch.float32).to(self.device) - mask_on_vec

        mask_on = torch.sigmoid(head_mask + tail_mask)
        mask_off = torch.ones((len(self.feat), 1), dtype=torch.float32).to(self.device) - mask_on

        mask_on_node_dict = self.get_masked_emb(mask_on)
        mask_off_node_dict = self.get_masked_emb(mask_off)

        mask_on_embed = self.model(self.graph, mask_on_node_dict)
        mask_off_embed = self.model(self.graph, mask_off_node_dict)

        return mask_on_embed['user'][head], mask_on_embed['item'][tail], mask_off_embed['user'][head], \
            mask_off_embed['item'][tail]
