import os
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import dgl
import copy


class HeteroDataset:
    def __init__(self):
        self.file_location = "dataset/"
        self.dataset_name = "movielens/"
        self.dir_to_load = self.file_location + self.dataset_name

        self.graph_filename = "movielen_e.csv"
        self.train_graph_file = "new_train.csv"
        self.test_graph_file = "new_test.csv"

        self.edge_label_filename = "./attributes_info/movielens_kg_label_map.csv"
        self.node_feature_filename = "./feature_vectors_v.csv"

        self.edge_labels = pd.read_csv(self.dir_to_load + self.edge_label_filename).to_numpy()
        self.node_features = pd.read_csv(self.dir_to_load + self.node_feature_filename).to_numpy()

        self.origin_graph = pd.read_csv(self.dir_to_load + self.graph_filename)
        self.train_graph = pd.read_csv(self.dir_to_load + self.train_graph_file)
        self.test_graph = pd.read_csv(self.dir_to_load + self.test_graph_file)

        self.graph = self.load_graph()

        self.item_num = 3706
        self.user_num = 6040
        self.kg_num = 21201 - self.item_num - self.user_num

        self.add_node_features()

    def load_graph(self):
        edges_group = self.origin_graph.groupby(['label_id:int'])
        hetero_graph_dict = {}
        for label_id in edges_group.groups:
            edges_of_id = edges_group.get_group(label_id)
            srcs = edges_of_id["source_id:int"].to_numpy()
            dsts = edges_of_id["target_id:int"].to_numpy()
            if label_id == 0 or label_id == 1:
                edge_type_name = ('user', str(self.edge_labels[label_id][1]), 'item')
            elif label_id == 2 or label_id == 3:
                edge_type_name = ('item', str(self.edge_labels[label_id][1]), 'user')
            elif 3 < label_id < 24:
                edge_type_name = ('item', str(self.edge_labels[label_id][1]), 'kg')
            else:
                edge_type_name = ('kg', str(self.edge_labels[label_id][1]), 'item')

            if edge_type_name in hetero_graph_dict.keys():
                print("already exist...")
            else:
                hetero_graph_dict[edge_type_name] = (torch.from_numpy(srcs), torch.from_numpy(dsts))

        return dgl.heterograph(hetero_graph_dict)

    def add_node_features(self):
        torch.manual_seed(3407)
        self.graph.nodes['item'].data['feat'] = torch.from_numpy(self.node_features[:3706, 1:])
        self.graph.nodes['user'].data['feat'] = torch.from_numpy(self.node_features[:, 1:])
        self.graph.nodes['kg'].data['feat'] = torch.cat(
            (self.graph.nodes['user'].data['feat'], torch.randn(self.kg_num, 16))).to(torch.float32)

    def get_train_batch(self, batch_size):
        # sampled_data = self.train_graph.sample(batch_size).values
        sampled_data = self.train_graph.sample(batch_size).values
        return sampled_data

    def get_test(self):
        return self.test_graph.values

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def get_kg_num(self):
        return self.kg_num
