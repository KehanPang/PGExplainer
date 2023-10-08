import torch

from dataset import *
from model import *
import argparse
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('-embed_dim', default=16, type=int, help="embedding dimension")
    parser.add_argument('-hidden_dim', default=48, type=int, help="embedding dimension")
    parser.add_argument('-reg_lambda', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-batch_size', default=5120, type=int, help="batch size")
    parser.add_argument('-epoch', default=100, type=int, help="batch size")
    parser.add_argument('-save_each', default=10, type=int, help="validate every k epochs")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parameter()
    dataset = HeteroDataset()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PinSAGE(16, args.hidden_dim, args.embed_dim, dataset.graph).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005,
                                 weight_decay=1e-6)  # Increased weight decay and reduced learning rate
    criterion = nn.BCEWithLogitsLoss()
    graph = copy.deepcopy(dataset.graph)

    n_features = 16

    h = []
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
        a = torch.index_select(h, dim=0, index=torch.tensor(first_elements).to(device))
        b = torch.index_select(h, dim=0, index=torch.tensor(second_elements).to(device))
        c = (a * b).sum(dim=1)
        ground_truth = 2 * torch.tensor(labels).to(device) - 1

        print(roc_auc_score(torch.tensor(labels).detach().to('cpu').numpy(), c.detach().to('cpu').numpy()))

        loss = criterion(c, ground_truth.to(torch.float32))
        # print(loss)
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

    print("===================Testing===========================")
    model.eval()
    test = dataset.get_test()
    first_elements, second_elements, labels = zip(*test)

    a = torch.index_select(h, dim=0, index=torch.tensor(first_elements).to(device))
    b = torch.index_select(h, dim=0, index=torch.tensor(second_elements).to(device))
    c = (a * b).sum(dim=1)
    # c = torch.where(c > 0, torch.tensor(1), torch.tensor(-1)).to('cpu')
    # ground_truth = 2 * torch.tensor(labels).to(device).to('cpu') - 1
    c = (c.to('cpu') + 1) / 2
    # c = torch.sigmoid(c).to('cpu')
    ground_truth = torch.tensor(labels).to(device).to('cpu')
    print(roc_auc_score(ground_truth.detach().numpy(), c.detach().numpy()))
