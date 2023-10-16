import torch

from HeteroDataset import *
from HomoDataset import *
from HeteroModel import *
import argparse
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from args import *

args = get_parameter()

directory = "models/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = directory + "{}_hetero.chkpnt".format(args.epoch)
model = torch.load(model_path, map_location=device)

print("===================Testing===========================")
model.eval()
dataset = HeteroDataset()
graph = copy.deepcopy(dataset.graph)
model.eval()
test = dataset.get_test()
first_elements, second_elements, labels = zip(*test)

node_dict = {node_type: graph.nodes[node_type].data['feat'].to(device).to(torch.float32) for node_type in
             graph.ntypes}
h = model(graph, node_dict)

item = h['item'].to(device)
user = h['user'].to(device)
kg = h['kg'].to(device)

feat = torch.cat((item, user[len(item):], kg[len(user):]))
feat = F.normalize(feat, p=2, dim=1)

a = torch.index_select(feat, dim=0, index=torch.tensor(first_elements).to(device))
b = torch.index_select(feat, dim=0, index=torch.tensor(second_elements).to(device))
c = (a * b).sum(dim=1).to('cpu')
c = (c.to('cpu') + 1) / 2

ground_truth = torch.tensor(labels).to('cpu')

print(roc_auc_score(ground_truth.detach().numpy(), c.detach().numpy()))
