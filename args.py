import argparse


def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', default=0.005, type=float, help="learning rate")
    parser.add_argument('-embed_dim', default=32, type=int, help="embedding dimension")
    parser.add_argument('-hidden_dim', default=64, type=int, help="embedding dimension")
    parser.add_argument('-weight_decay', default=1e-6, type=float, help="l2 regularization parameter")
    parser.add_argument('-batch_size', default=5120, type=int, help="batch size")
    parser.add_argument('-epoch', default=100, type=int, help="batch size")
    args = parser.parse_args()
    return args
