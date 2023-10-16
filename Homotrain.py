import torch
from HeteroDataset import *
from HomoDataset import *
from HeteroModel import *
from HomoModel import *
import argparse
import copy
from sklearn.metrics import f1_score


from args import *


args = get_parameter()
dataset = HomoDataset()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(16, args.hidden_dim, args.embed_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05,
                             weight_decay=5e-4)  # Increased weight decay and reduced learning rate
criterion = nn.BCEWithLogitsLoss()
graph = copy.deepcopy(dataset.graph)
n_features = graph.ndata['h'].shape[1]
k = 5
opt = torch.optim.Adam(model.parameters())

h = []
model.train()
print("===================Training==========================")
for epoch in range(args.epoch):
    h = model(graph, graph.ndata['h'])
    samples = dataset.get_train_batch(args.batch_size)
    first_elements, second_elements, labels = zip(*samples)
    a = torch.index_select(h, dim=0, index=torch.tensor(first_elements).to(device))
    b = torch.index_select(h, dim=0, index=torch.tensor(second_elements).to(device))
    ground_truth = 2 * torch.tensor(labels).to(device) - 1

    loss = criterion((a * b).sum(dim=1), ground_truth.to(torch.float32))
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新模型参数

print("Saving the model")
directory = "models/"
if not os.path.exists(directory):
    os.makedirs(directory)
torch.save(model, directory + str(args.epoch) + "_homo.chkpnt")
