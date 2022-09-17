#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import argparse
from earlystopping import EarlyStopping
import torch
import torch.optim as optim
import copy
import os
from torch_geometric.loader import NeighborLoader
print("before model")
from models import GCN, GAT, SAGE_sup, MLP, H2GCN
print("after model")
from metrics import f1_loss, BCE_loss, _eval_rocauc, ap_score
print("imported")
from torch_geometric.data import Data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default='Hyperspheres_10_10_0',
                        help='Name of the dataset:'
                             'blogcatalog'
                             'Hyperspheres_64_64_0'
                             'ogbn-proteins'
                             'pcg_removed_isolated_nodes'
                             'Humloc')
    parser.add_argument('--train_percent', type=float, default=0.6,
                        help='percentage of data used for training')
    parser.add_argument("--split_name", default='split_2.pt',
                        help='Name of the split')
    parser.add_argument("--model_name", default='H2GCN',
                        help='GCN, GAT, SAGE_sup, MLP, H2GCN')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument("--device_name", default='cuda',
                        help='Name of the device')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--patience', type=float, default=100,
                        help='patience for early stopping.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='patience for early stopping.')
    return parser.parse_args()


def model_train():
    model.train()
    optimizer.zero_grad()
    print(model)
    if args.model_name == "MLP":
        output = model(x)
    else:
        output = model(x, edge_index)
    loss_train = BCE_loss(output[train_mask], labels[train_mask])

    micro_train, macro_train = f1_loss(labels[train_mask], output[train_mask])
    #roc_auc_train_micro, roc_auc_train_macro = _eval_rocauc(labels[train_mask], output[train_mask])
    roc_auc_train_macro = _eval_rocauc(labels[train_mask], output[train_mask])

    loss_train.backward()
    optimizer.step()

    return loss_train, micro_train, macro_train, roc_auc_train_macro
           #roc_auc_train_micro, \



@torch.no_grad()
def model_test():
    model.eval()
    if args.model_name == "MLP":
        output = model(x)
    #elif args.model_name == "SAGE_sup":
    #    output = model.full_forward(x, edge_index)
    else:
        output = model(x, edge_index)

    loss_val = BCE_loss(output[val_mask], labels[val_mask])

    micro_val, macro_val = f1_loss(labels[val_mask], output[val_mask])

    #roc_auc_val_micro, roc_auc_val_macro = _eval_rocauc(labels[val_mask], output[val_mask])
    roc_auc_val_macro = _eval_rocauc(labels[val_mask], output[val_mask])

    micro_test, macro_test = f1_loss(labels[test_mask], output[test_mask])
    #roc_auc_test_micro, roc_auc_test_macro = _eval_rocauc(labels[test_mask], output[test_mask])
    roc_auc_test_macro = _eval_rocauc(labels[test_mask], output[test_mask])

    ap_test = ap_score(labels[test_mask], output[test_mask])

    return loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, ap_test


def batch_train(loader):
    total_loss = 0
    model.train()

    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        labels_target = batch.y[:batch.batch_size]
        loss = BCE_loss(out, labels_target)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / torch.sum(G.train_mask)


@torch.no_grad()
def batch_test(subgraph_loader):

    output = model.inference(G.x, subgraph_loader)
    loss_val = BCE_loss(output[val_mask], labels[val_mask])
    micro_val, macro_val = f1_loss(labels[val_mask], output[val_mask])
    #roc_auc_val_micro, roc_auc_val_macro = _eval_rocauc(labels[val_mask], output[val_mask])
    roc_auc_val_macro = _eval_rocauc(labels[val_mask], output[val_mask])

    micro_test, macro_test = f1_loss(labels[test_mask], output[test_mask])
    #roc_auc_test_micro, roc_auc_test_macro = _eval_rocauc(labels[test_mask], output[test_mask])
    roc_auc_test_macro = _eval_rocauc(labels[test_mask], output[test_mask])

    ap_test = ap_score(labels[test_mask], output[test_mask])

    #return loss_val, micro_val, macro_val, roc_auc_val_micro, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_micro, roc_auc_test_macro, ap_test
    return loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, ap_test

# @torch.no_grad()
# def batch_test(loader):
#     output = []
#     for batch_size, n_id, adjs in loader:
#         adjs = [adj.to(device) for adj in adjs]
#         out = model(x[n_id], adjs)
#         output.append(out)
#     output = torch.cat(output, dim=0)
#     loss_val = BCE_loss(output[val_mask], labels[val_mask])
#     micro_val, macro_val = f1_loss(labels[val_mask], output[val_mask])
#     #roc_auc_val_micro, roc_auc_val_macro = _eval_rocauc(labels[val_mask], output[val_mask])
#     roc_auc_val_macro = _eval_rocauc(labels[val_mask], output[val_mask])
#
#     micro_test, macro_test = f1_loss(labels[test_mask], output[test_mask])
#     #roc_auc_test_micro, roc_auc_test_macro = _eval_rocauc(labels[test_mask], output[test_mask])
#     roc_auc_test_macro = _eval_rocauc(labels[test_mask], output[test_mask])
#
#     ap_test = ap_score(labels[test_mask], output[test_mask])
#
#     #return loss_val, micro_val, macro_val, roc_auc_val_micro, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_micro, roc_auc_test_macro, ap_test
#     return loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, ap_test


def load_hyper_data(data_name, split_name, train_percent, path="../../../data/"):
    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           skip_header=1, dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            skip_header=1, dtype=np.dtype(float), delimiter=',')).float()
    #features = features[:, 0:]
    print(features.shape)
    #features = torch.randn(size=[labels.shape[0], 128])

    edges = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edges.txt"),
                         dtype=np.dtype(float), delimiter=',')).long()
    edge_index = torch.transpose(edges, 0, 1)
    print(edge_index.shape)

    folder_name = data_name + "_" + str(train_percent)
    file_path = os.path.join(path, folder_name, split_name)

    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    y = labels.clone().detach().float()
    y[val_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])
    y[test_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])

    #num_class = labels.shape[1]
    num_nodes = labels.shape[0]

    # cnt_id = torch.unique(torch.flatten(edge_index)).tolist()
    # iso_mask = torch.ones(features.shape[0], dtype=torch.bool)
    # iso_mask[cnt_id] = False

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.soft_labels = y
    G.n_id = torch.arange(num_nodes)

    #return adj, edge_index, features, labels, y, train_mask, val_mask, test_mask, num_class, num_nodes #, iso_mask
    return G


if __name__ == "__main__":

    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device = args.device_name

    #if args.data_name in ["blogcatalog", "flickr", "youtube"]:
    #    G = load_mat_data(args.data_name, args.split_name, args.train_percent)

    #elif args.data_name == "pcg_removed_isolated_nodes":
    #    G = load_pcg(args.data_name, args.split_name, args.train_percent)

    if args.data_name.startswith("Hypersphere"):
        G = load_hyper_data(args.data_name, args.split_name, args.train_percent)

    # elif args.data_name == "yelp":
    #     G = import_yelp(args.data_name, args.split_name, args.train_percent)
    #
    # elif args.data_name == "ogbn-proteins":
    #     G = import_ogb(args.data_name)
    #
    # elif args.data_name == "Humloc":
    #     G = load_humloc()
    #
    # elif args.data_name == "Eukloc":
    #     G = load_eukloc()

    else:
        raise OSError("Dataset not found")

    if args.model_name == "GCN":
        model = GCN(in_channels=G.x.shape[1],
                    hidden_channels=args.hidden,
                    class_channels=G.y.shape[1],
                    )

    elif args.model_name == "SAGE_sup":
        model = SAGE_sup(in_channels=G.x.shape[1],
                         hidden_channels=args.hidden,
                         class_channels=G.y.shape[1])
        train_loader = NeighborLoader(G,
                                      input_nodes=G.train_mask,
                                      num_neighbors=[-1, -1],
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      )

        # val_loader = NeighborSampler(copy.copy(G.edge_index),
        #                              sizes=[-1, -1],
        #                              node_idx=G.val_mask,
        #                              batch_size=8,
        #                              shuffle=False,
        #                              num_nodes=G.num_nodes
        #                              )
        subgraph_loader = NeighborLoader(copy.copy(G),
                                         input_nodes=None,
                                         num_neighbors=[-1],
                                         shuffle=False,
                                         batch_size=64)
        # subgraph_loader = NeighborSampler(copy.copy(G.edge_index),
        #                                   sizes=[-1, -1],
        #                                   batch_size=8,
        #                                   shuffle=False,
        #                                   num_nodes=G.num_nodes
        #                                   )

    elif args.model_name == "GAT":
        model = GAT(in_channels=G.x.shape[1],
                    class_channels=G.y.shape[1])

    elif args.model_name == "MLP":
        model = MLP(in_channels=G.x.shape[1],
                    hidden_channels=args.hidden,
                    class_channels=G.y.shape[1])

    elif args.model_name == "H2GCN":
        model = H2GCN(nfeat=G.x.shape[1],
                      nhid=args.hidden,
                      nclass=G.y.shape[1])

    else:
        raise OSError("model not defined")

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay
                           )

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    model.to(device)
    x = G.x.to(device)
    labels = G.y.to(device)
    edge_index = G.edge_index.to(device)
    train_mask = G.train_mask.to(device)
    val_mask = G.val_mask.to(device)
    test_mask = G.test_mask.to(device)

    if args.model_name == "SAGE_sup":
        for epoch in range(1, args.epochs):
            loss_train = batch_train(train_loader)
            #loss_val, model = batch_val(val_loader)

            #_, micro_val, macro_val, roc_auc_val_micro, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_micro, roc_auc_test_macro, test_ap = batch_test(subgraph_loader)
            loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = batch_test(subgraph_loader)
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                  f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f} '
                  f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
                  #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
                  f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                  #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
                  f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                  f'Test Average Precision Score: {test_ap:.4f}, '
                  )
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Optimization Finished!")
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))
        #_, micro_val, macro_val, roc_auc_val_micro, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_micro, roc_auc_test_macro, test_ap = batch_test(subgraph_loader)
        _, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = batch_test(subgraph_loader)
        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
              #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
              f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
              f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test Average Precision Score: {test_ap:.4f}, '
              )
    # other models
    else:
        for epoch in range(1, args.epochs):
            #loss_train, micro_train, macro_train, roc_auc_train_micro, roc_auc_train_macro = model_train()
            loss_train, micro_train, macro_train, roc_auc_train_macro = model_train()
            #loss_val, micro_val, macro_val, roc_auc_val_micro, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_micro, roc_auc_test_macro, test_ap = model_test()
            loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()

            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                  f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f} '
                  f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f} '
                  f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
                  #f'Train ROC-AUC micro: {roc_auc_train_micro:.4f}, '
                  f'train ROC-AUC macro: {roc_auc_train_macro:.4f} '
                  #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
                  f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                  #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
                  f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                  f'Test Average Precision Score: {test_ap:.4f}, '
                  )
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Optimization Finished!")
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))
        #loss_val, micro_val, macro_val, roc_auc_val_micro, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_micro, roc_auc_test_macro, test_ap = model_test()
        loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()
        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
              #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
              f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
              f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test Average Precision Score: {test_ap:.4f}, '
             )


#print(hp.heap())






