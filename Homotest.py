import torch
from HeteroDataset import *
from HomoDataset import *
from HeteroModel import *
import argparse
import copy
from sklearn.metrics import f1_score


def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('-embed_dim', default=32, type=int, help="embedding dimension")
    parser.add_argument('-hidden_dim', default=64, type=int, help="embedding dimension")
    parser.add_argument('-reg_lambda', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-batch_size', default=500, type=int, help="batch size")
    parser.add_argument('-epoch', default=100, type=int, help="batch size")
    parser.add_argument('-save_each', default=10, type=int, help="validate every k epochs")
    args = parser.parse_args()
    return args


args = get_parameter()
directory = "models/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = directory + str(args.epoch) + "_homo.chkpnt"

model = torch.load(model_path, map_location=device)

print("===================Testing===========================")
dataset = HomoDataset()
graph = copy.deepcopy(dataset.graph)
model.eval()
test = dataset.get_test()
first_elements, second_elements, labels = zip(*test)
h = model(graph, graph.ndata['h'])
a = torch.index_select(h, dim=0, index=torch.tensor(first_elements).to(device))
b = torch.index_select(h, dim=0, index=torch.tensor(second_elements).to(device))
c = (a * b).sum(dim=1)
# c = torch.where(c > 0, torch.tensor(1), torch.tensor(-1)).to('cpu')
# ground_truth = 2 * torch.tensor(labels).to(device).to('cpu') - 1
# print(f1_score(ground_truth, c))
c = (c.to('cpu')) + 1 / 2
ground_truth = torch.tensor(labels).to(device).to('cpu')
print(roc_auc_score(ground_truth.detach().numpy(), c.detach().numpy()))