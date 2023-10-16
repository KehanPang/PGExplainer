import os
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import dgl
import copy


class HomoDataset:
    def __init__(self):
        self.file_location = "dataset/"
        self.dataset_name = "movielens/"
        self.dir_to_load = self.file_location + self.dataset_name

        self.graph_filename = "movielen_e.csv"
        self.train_graph_file = "movielen_train.csv"
        self.test_graph_file = "movielen_test.csv"

        self.edge_label_filename = "./attributes_info/movielens_kg_label_map.csv"
        self.node_feature_filename = "./feature_vectors_v.csv"

        self.edge_labels = pd.read_csv(self.dir_to_load + self.edge_label_filename).to_numpy()
        self.node_features = pd.read_csv(self.dir_to_load + self.node_feature_filename).to_numpy()

        self.origin_graph = pd.read_csv(self.dir_to_load + self.graph_filename)
        self.train_graph = pd.read_csv(self.dir_to_load + self.train_graph_file)
        self.test_graph = pd.read_csv(self.dir_to_load + self.test_graph_file)

        self.sources = []
        self.sinks = []
        self.rels = []
        self.rels_num = 0

        self.graph = self.load_graph()

        self.item_num = 3706
        self.user_num = 6040
        self.kg_num = 21201 - self.item_num - self.user_num
        self.add_node_features()

    def load_graph(self):
        self.sources = self.origin_graph.iloc[:, 1].values
        self.sinks = self.origin_graph.iloc[:, 2].values
        self.rels = self.origin_graph.iloc[:, 3].values
        self.rels_num = len(set(self.rels))
        graph = dgl.graph((self.sources, self.sinks))
        return graph

    def add_node_features(self):
        self.graph.ndata['h'] = torch.cat(
            (torch.from_numpy(self.node_features[:, 1:]), torch.randn(self.kg_num, 16))).to(torch.float32)

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
