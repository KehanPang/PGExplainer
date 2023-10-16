import torch

from HeteroDataset import *
from HomoDataset import *
from HeteroModel import *
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from args import *

if __name__ == '__main__':
    args = get_parameter()
    dataset = HeteroDataset()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PinSAGE(16, args.hidden_dim, args.embed_dim, device, dataset.graph).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)  # Increased weight decay and reduced learning rate
    criterion = nn.BCEWithLogitsLoss()
    graph = copy.deepcopy(dataset.graph)

    n_features = 16

    h = {}
    user = []
    item = []
    model.train()
    print("===================Training==========================")
    for epoch in range(args.epoch):
        node_dict = {node_type: graph.nodes[node_type].data['feat'].to(device).to(torch.float32) for node_type in
                     graph.ntypes}
        h = model(graph, node_dict)
        samples = dataset.get_train_batch(args.batch_size)
        first_elements, second_elements, labels = zip(*samples)

        item = h['item'].to(device)
        user = h['user'].to(device)
        kg = h['kg'].to(device)

        feat = torch.cat((item, user[len(item):], kg[len(user):]))
        feat = F.normalize(feat, p=2, dim=1)

        a = torch.index_select(feat, dim=0, index=torch.tensor(first_elements).to(device))
        b = torch.index_select(feat, dim=0, index=torch.tensor(second_elements).to(device))
        c = (a * b).sum(dim=1).to(device)

        ground_truth = 2 * torch.tensor(labels).to(device) - 1

        print(roc_auc_score(torch.tensor(labels).detach().to('cpu').numpy(), ((c.detach().to('cpu') + 1) / 2).numpy()))

        loss = criterion(c, ground_truth.to(torch.float32))
        # print(loss)
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

    print("===================Testing===========================")
    model.eval()
    test = dataset.get_test()
    first_elements, second_elements, labels = zip(*test)

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
