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

args = get_parameter()

print("===================Loading GNN Model===========================")
directory = "models/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = directory + "5000_hetero.chkpnt"
gnn_model = torch.load(model_path, map_location=device)
for param in gnn_model.parameters():
    param.requires_grad = False
gnn_model.eval()
dataset = HeteroDataset()
graph = copy.deepcopy(dataset.graph)
node_dict = {node_type: graph.nodes[node_type].data['feat'].to(device).to(torch.float32) for node_type in
             graph.ntypes}
embed = gnn_model(graph, node_dict)
item = embed['item'].to(device)
user = embed['user'].to(device)
kg = embed['kg'].to(device)
feat = torch.cat((item, user[len(item):], kg[len(user):]))
# feat = F.normalize(feat, p=2, dim=1)

explainer = PGExplainer(gnn_model, args.embed_dim, args.hidden_dim, device, graph, feat)
optimizer = torch.optim.Adam(explainer.parameters(), lr=0.00005,
                             weight_decay=1e-5)  # Increased weight decay and reduced learning rate

exp_batch = 5120
print("===================Training EXP Model===========================")
explainer.train()
exp_test_graph = pd.read_csv("dataset/movielens/test_pos.csv")
for i in range(args.epoch):
    samples = exp_test_graph.sample(exp_batch).values
    for item in samples:
        optimizer.zero_grad()
        head, tail, label = item
        head_emb_mask_on, tail_emb_mask_on, head_emb_mask_off, tail_emb_mask_off = explainer(head, tail)

        raw_head = feat[head]
        raw_tail = feat[tail]

        raw_pred = (raw_head * raw_tail).sum(dim=-1)
        mask_on_pred = (head_emb_mask_on * tail_emb_mask_on).sum(dim=-1)
        mask_off_pred = (head_emb_mask_off * tail_emb_mask_off).sum(dim=-1)

        c1 = nn.SmoothL1Loss()
        c2 = nn.MSELoss()

        weight = 0.95

        loss = weight * c1(raw_pred, mask_on_pred) + (weight - 1) * c1(
            mask_off_pred, mask_on_pred)

        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

    print("Epoch {}, Saving the model....".format(i))
    directory = "exp/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(explainer, directory + "{}_{}_exp.chkpnt".format(i, exp_batch))
    print("Saving success")
