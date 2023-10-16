import torch
from HeteroDataset import *
from HomoDataset import *
from HeteroModel import *
import argparse
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from pgexplainer import *
from args import *

print("===================Loading EXP Model===========================")
directory = "exp/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = directory + "85_5120_exp.chkpnt"
explainer = torch.load(model_path, map_location=device)
explainer.eval()
exp_test_graph = pd.read_csv("dataset/movielens/test_pos.csv")

feat = explainer.feat

with open("result.csv", "w") as f:
    for item in exp_test_graph.values:
        head, tail, label = item
        head_emb_mask_on, tail_emb_mask_on, head_emb_mask_off, tail_emb_mask_off = explainer(head, tail)

        raw_head = feat[head]
        raw_tail = feat[tail]

        raw_pred = (raw_head * raw_tail).sum(dim=-1)

        mask_on_pred = (head_emb_mask_on * tail_emb_mask_on).sum(dim=-1)
        mask_off_pred = (head_emb_mask_off * tail_emb_mask_off).sum(dim=-1)

        gnn_pred = torch.sigmoid(raw_pred).to('cpu')
        exp_pred = torch.sigmoid(mask_on_pred).to('cpu')

        f.write(
            "{},{},{},{}\n".format(head, tail, torch.round(gnn_pred * 1000) / 1000,
                                   torch.round(exp_pred * 1000) / 1000))
