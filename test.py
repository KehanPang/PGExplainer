import torch

from dataset import *
from model import *
import argparse
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


directory = "models/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = directory + "5000_hetero.chkpnt"
model = torch.load(model_path, map_location=device)

print("===================Testing===========================")
model.eval()
dataset = HeteroDataset()
graph = copy.deepcopy(dataset.graph)
test = dataset.get_test()
first_elements, second_elements, labels = zip(*test)
node_dict = {node_type: graph.nodes[node_type].data['feat'].to(device).to(torch.float32) for node_type in
             graph.ntypes}
h = model(graph, node_dict)
a = torch.index_select(h, dim=0, index=torch.tensor(first_elements).to(device))
b = torch.index_select(h, dim=0, index=torch.tensor(second_elements).to(device))
c = (a * b).sum(dim=1)
# c = torch.where(c > 0, torch.tensor(1), torch.tensor(-1)).to('cpu')
# ground_truth = 2 * torch.tensor(labels).to(device).to('cpu') - 1
c = (c.to('cpu') + 1) / 2
ground_truth = torch.tensor(labels).to(device).to('cpu')
print(roc_auc_score(ground_truth.detach().numpy(), c.detach().numpy()))
