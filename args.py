import argparse

def add_default_args():
    parser = argparse.ArgumentParser(description='DeepRDR')

    parser.add_argument('--seed', type=int, default=123, help="seed")
    parser.add_argument('--task', type=str, default="pretrain", help="task: pretrain, test1, test2")
    parser.add_argument('--fold', type=int, default=1, help="fold")
    parser.add_argument('--pretrain', type=str, default="", help="path to pretrained model")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size")
    parser.add_argument('--epoch_num', type=int, default=100, help="epoch number")
    parser.add_argument('--h_dim', type=int, default=512, help="hidden dimension")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--adj_loss_coef', type=float, default=5e-4, help="adj_loss_coef")
    parser.add_argument('--param_l2_coef', type=float, default=5e-4, help="param_l2_coef")
    parser.add_argument('--drug_dim', type=int, default=820, help="drug dimension")
    parser.add_argument('--mesh_dim', type=int, default=2495, help="mesh dimension")

    return parser