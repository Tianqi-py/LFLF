#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import argparse
from earlystopping import EarlyStopping
import torch
import torch.optim as optim
import copy
# for different version of Py_G
#from torch_geometric.data import NeighborSampler as RawNeighborSampler
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_cluster import random_walk
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from torch_geometric.loader import NeighborLoader

from metrics import _eval_rocauc, f1_loss, ap_score
from utils import load_pcg, load_mat_data, load_hyper_data, import_yelp, load_humloc, load_eukloc, import_ogb
from LFLF_unsup_models import LFLF_GCN, LFLF_SAGE, LFLF_GAT
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default='Hyperspheres_10_10_0',
                        help='Name of the dataset'
                        'Hyperspheres_64_64_0'
                        'pcg_removed_isolated_nodes'
                        'Humloc')
    parser.add_argument("--edge_name",
                        default='a=0.3_b=0.25_homo=0.23935293259574536.txt',
                        help='Name of the edges:'
                             'a=0.3_b=0.25_homo=0.23935293259574536.txt'
                             'a=5_b=0.09_homo=0.4376710132277844.txt'
                             'a=6_b=0.07_homo=0.5921203716773683.txt'
                             'a=7_b=0.04_homo=0.85598313310031.txt'
                             'a=8_b=0.02_homo=0.9991270191292244.txt'

                        )
    parser.add_argument('--train_percent', type=float, default=0.6,
                        help='percentage of data used for training')
    parser.add_argument("--split_name", default='split_0.pt',
                        help='Name of the split')
    parser.add_argument("--model_name", default='LFLF_SAGE',
                        help='LFLF_GCN or LFLF_GAT or LFLF_SAGE')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument("--device_name", default='cuda',
                        help='Name of the device')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--layer', type=float, default=2,
                        help='number of layer in LFLF')
    parser.add_argument('--patience', type=float, default=100,
                        help='patience for early stopping.')
    parser.add_argument('--num_walks', type=int, default=20,
                        help='number of positive neighbors')
    parser.add_argument('--pos_neg_ratio', type=int, default=3,
                        help='the ratio of positive neighbors to negative neighbors')
    parser.add_argument('--sample_per_layer', type=list, default=[25, 10],
                        help='neighbor sample per layer for aggregation')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='patience for early stopping.')
    return parser.parse_args()


class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        original_batch = batch
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample num_walks direct neighbor (as positive
        # example) and num_walks random node (as negative example):

        for i in torch.arange(args.num_walks):
            pos_batch = random_walk(row, col, original_batch, walk_length=1,
                                    coalesced=False)[:, 1]
            batch = torch.cat([batch, pos_batch], dim=0)
        for j in torch.arange(args.num_walks * args.pos_neg_ratio):
            neg_batch = torch.randint(0, self.adj_t.size(0), (original_batch.numel(), ),
                                      dtype=torch.long)
            batch = torch.cat([batch, neg_batch], dim=0)

        return super().sample(batch)


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


def model_train(train_loader):
    total_loss = 0
    model.train()

    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        edge_weights = []
        for i, (edge_index, e_id, size) in enumerate(adjs):
            edge_weights.append(G.edge_attr[e_id].to(device))
        out, _ = model(x[n_id], soft_labels[n_id], adjs, edge_weights)

        # original batch, po_batch, neg_batch
        num_splits = 1 + args.num_walks + args.pos_neg_ratio * args.num_walks
        out_target = out.split(out.size(0) // num_splits, dim=0)[0]
        pos_out = torch.vstack(out.split(out.size(0) // num_splits, dim=0)[1:args.num_walks+1])
        neg_out = torch.vstack(out.split(out.size(0) // num_splits, dim=0)[args.num_walks+1:])

        out_repeat_pos = torch.cat(args.num_walks * [out_target])
        out_repeat_neg = torch.cat(args.num_walks * args.pos_neg_ratio * [out_target])

        pos_loss = F.logsigmoid((out_repeat_pos * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out_repeat_neg * neg_out).sum(-1)).mean()
        loss = - pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out_target.size(0)

    return total_loss / G.num_nodes


@torch.no_grad()
def model_val(val_loader):
    total_loss_val = 0
    model.eval()
    for batch_size, n_id, adjs in val_loader:
        adjs = [adj.to(device) for adj in adjs]
        edge_weights = []
        for i, (edge_index, e_id, size) in enumerate(adjs):
            edge_weights.append(G.edge_attr[e_id].to(device))

        out, _ = model(x[n_id], soft_labels[n_id], adjs, edge_weights)

        num_splits = 1 + args.num_walks + args.pos_neg_ratio * args.num_walks
        out_target = out.split(out.size(0) // num_splits, dim=0)[0]
        #val_mask_in_batch = val_mask[n_id[:out_target.shape[0]]]
        # there is validation data in current batch
        #if torch.sum(val_mask_in_batch):

        #val_data = out_target[val_mask_in_batch]

        pos_out = torch.vstack(out.split(out.size(0) // num_splits, dim=0)[1:args.num_walks + 1])
        neg_out = torch.vstack(out.split(out.size(0) // num_splits, dim=0)[args.num_walks + 1:])

        #val_mask_pos_repeat = torch.cat(args.num_walks * [val_mask_in_batch])
        #val_mask_neg_repeat = torch.cat(args.pos_neg_ratio * args.num_walks * [val_mask_in_batch])

        #val_mask_pos_repeat = torch.cat(args.num_walks * [val_mask_in_batch])
        #val_mask_neg_repeat = torch.cat(args.pos_neg_ratio * args.num_walks * [val_mask_in_batch])

        #pos_out_val = pos_out[val_mask_pos_repeat]
        #neg_out_val = neg_out[val_mask_neg_repeat]

        val_data_repeat_pos = torch.cat(args.num_walks * [out_target])
        val_data_repeat_neg = torch.cat(args.pos_neg_ratio * args.num_walks * [out_target])

        #pos_loss_val = F.logsigmoid((val_data_repeat_pos * pos_out_val).sum(-1)).mean()
        #neg_loss_val = F.logsigmoid(-(val_data_repeat_neg * neg_out_val).sum(-1)).mean()
        pos_loss_val = F.logsigmoid((val_data_repeat_pos * pos_out).sum(-1)).mean()
        neg_loss_val = F.logsigmoid(-(val_data_repeat_neg * neg_out).sum(-1)).mean()
        loss_val = -pos_loss_val - neg_loss_val
        #total_loss_val += float(loss_val) * val_data.size(0)
        total_loss_val += float(loss_val) * out_target.size(0)
        # # no validation data in the current batch
        # else:
        #     continue
    return total_loss_val / (torch.sum(G.val_mask)), model


@torch.no_grad()
def model_test():
    model.eval()
    output, _ = model.full_forward(x, soft_labels, edge_index_full_graph, edge_attr)
    print(output.shape)
    #################################################
    # logistic regression
    feature_matrix = np.asarray(output, dtype=np.float64)
    labels_matrix = labels.cpu().numpy()

    X_train = feature_matrix[train_mask.cpu()]
    y_train_ = labels_matrix[train_mask.cpu()]
    y_train = [[] for x in range(y_train_.shape[0])]

    row, col = np.nonzero(y_train_)
    for i, j in zip(row, col):
        y_train[i].append(j)

    X_test = feature_matrix[test_mask.cpu()]
    y_test_ = labels_matrix[test_mask.cpu()]
    y_test = [[] for _ in range(y_test_.shape[0])]

    row, col = np.nonzero(y_test_)
    for i, j in zip(row, col):
        y_test[i].append(j)

    X_val = feature_matrix[val_mask.cpu()]
    y_val_ = labels_matrix[val_mask.cpu()]
    y_val = [[] for _ in range(y_val_.shape[0])]

    row, col = np.nonzero(y_val_)
    for i, j in zip(row, col):
        y_val[i].append(j)

    clf = TopKRanker(LogisticRegression())
    clf.fit(X_train, y_train_)

    y_test_tensor = torch.from_numpy(y_test_)
    # find out how many labels should be predicted
    top_k_list = [len(l) for l in y_test]
    preds = clf.predict(X_test, top_k_list)

    y_train_tensor = torch.from_numpy(y_train_)
    top_list = [len(l) for l in y_train]
    preds_train = clf.predict(X_train, top_list)

    y_val_tensor = torch.from_numpy(y_val_)
    top_list_val = [len(l) for l in y_val]
    preds_val = clf.predict(X_val, top_list_val)

    mlb = MultiLabelBinarizer(classes=range(labels.shape[1]))

    micro_test = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average="micro")
    macro_test = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average="macro")

    micro_val = f1_score(mlb.fit_transform(y_val), mlb.fit_transform(preds_val), average="micro")
    macro_val = f1_score(mlb.fit_transform(y_val), mlb.fit_transform(preds_val), average="macro")

    micro_train = f1_score(mlb.fit_transform(y_train), mlb.fit_transform(preds_train), average="micro")
    macro_train = f1_score(mlb.fit_transform(y_train), mlb.fit_transform(preds_train), average="macro")
    ##########################################################################################

    clf_pre_train = clf.predict_proba(X_train)
    clf_pre_val = clf.predict_proba(X_val)
    clf_pre_test = clf.predict_proba(X_test)

    # roc_auc_test_micro, roc_auc_test_macro = _eval_rocauc(y_test_tensor, clf_pre_test)
    # roc_auc_val_micro, roc_auc_val_macro = _eval_rocauc(y_val_tensor, clf_pre_val)
    # roc_auc_train_micro, roc_auc_train_macro = _eval_rocauc(y_train_tensor, clf_pre_train)
    roc_auc_test_macro = _eval_rocauc(y_test_tensor, clf_pre_test)
    roc_auc_val_macro = _eval_rocauc(y_val_tensor, clf_pre_val)
    roc_auc_train_macro = _eval_rocauc(y_train_tensor, clf_pre_train)

    ap_test = ap_score(y_test_tensor, clf_pre_test)

    return micro_train, macro_train, micro_test, macro_test, micro_val, macro_val, \
           roc_auc_train_macro, roc_auc_test_macro, roc_auc_val_macro, ap_test


@torch.no_grad()
def batch_test(subgraph_loader):

    ######################################################################
    # older loader
    # model.eval()
    # output = []
    #
    # for batch_size, n_id, adjs in subgraph_loader:
    #     adjs = [adj.to(device) for adj in adjs]
    #     edge_weights = []
    #     for i, (edge_index, e_id, size) in enumerate(adjs):
    #         edge_weights.append(G.edge_attr[e_id].to(device))
    #
    #     out, _ = model(x[n_id], soft_labels[n_id], adjs, edge_weights)
    #     #out, _ = model(x[], soft_labels, edge_index_full_graph, edge_attr)
    #     #num_splits = 1 + args.num_walks + args.pos_neg_ratio * args.num_walks
    #     #out = out.split(out.size(0) // num_splits, dim=0)[0]
    #     output.append(out)
    #
    # output = torch.cat(output, dim=0).cpu()
    #################################################
    output, _ = model.inference(x, soft_labels, subgraph_loader)

    # logistic regression
    feature_matrix = np.asarray(output.cpu(), dtype=np.float64)
    labels_matrix = labels.cpu().numpy()

    X_train = feature_matrix[train_mask.cpu()]
    y_train_ = labels_matrix[train_mask.cpu()]
    y_train = [[] for x in range(y_train_.shape[0])]

    row, col = np.nonzero(y_train_)
    for i, j in zip(row, col):
        y_train[i].append(j)

    X_test = feature_matrix[test_mask.cpu()]
    y_test_ = labels_matrix[test_mask.cpu()]
    y_test = [[] for _ in range(y_test_.shape[0])]

    row, col = np.nonzero(y_test_)
    for i, j in zip(row, col):
        y_test[i].append(j)

    X_val = feature_matrix[val_mask.cpu()]
    y_val_ = labels_matrix[val_mask.cpu()]
    y_val = [[] for _ in range(y_val_.shape[0])]

    row, col = np.nonzero(y_val_)
    for i, j in zip(row, col):
        y_val[i].append(j)

    clf = TopKRanker(LogisticRegression())
    clf.fit(X_train, y_train_)

    y_test_tensor = torch.from_numpy(y_test_)
    # find out how many labels should be predicted
    top_k_list = [len(l) for l in y_test]
    preds = clf.predict(X_test, top_k_list)

    y_train_tensor = torch.from_numpy(y_train_)
    top_list = [len(l) for l in y_train]
    preds_train = clf.predict(X_train, top_list)

    y_val_tensor = torch.from_numpy(y_val_)
    top_list_val = [len(l) for l in y_val]
    preds_val = clf.predict(X_val, top_list_val)

    mlb = MultiLabelBinarizer(classes=range(labels.shape[1]))

    micro_test = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average="micro")
    macro_test = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average="macro")

    micro_val = f1_score(mlb.fit_transform(y_val), mlb.fit_transform(preds_val), average="micro")
    macro_val = f1_score(mlb.fit_transform(y_val), mlb.fit_transform(preds_val), average="macro")

    micro_train = f1_score(mlb.fit_transform(y_train), mlb.fit_transform(preds_train), average="micro")
    macro_train = f1_score(mlb.fit_transform(y_train), mlb.fit_transform(preds_train), average="macro")
    ##########################################################################################

    clf_pre_train = clf.predict_proba(X_train)
    clf_pre_val = clf.predict_proba(X_val)
    clf_pre_test = clf.predict_proba(X_test)

    # roc_auc_test_micro, roc_auc_test_macro = _eval_rocauc(y_test_tensor, clf_pre_test)
    # roc_auc_val_micro, roc_auc_val_macro = _eval_rocauc(y_val_tensor, clf_pre_val)
    # roc_auc_train_micro, roc_auc_train_macro = _eval_rocauc(y_train_tensor, clf_pre_train)
    roc_auc_test_macro = _eval_rocauc(y_test_tensor, clf_pre_test)
    roc_auc_val_macro = _eval_rocauc(y_val_tensor, clf_pre_val)
    roc_auc_train_macro = _eval_rocauc(y_train_tensor, clf_pre_train)

    ap_test = ap_score(y_test_tensor, clf_pre_test)

    return micro_train, macro_train, micro_test, macro_test, micro_val, macro_val, \
           roc_auc_train_macro, roc_auc_test_macro, roc_auc_val_macro, ap_test


if __name__ == "__main__":

    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device = args.device_name

    if args.data_name in ["blogcatalog", "flickr", "youtube"]:
        G = load_mat_data(args.data_name, args.split_name, args.train_percent)

    elif args.data_name == "pcg_removed_isolated_nodes":
        G = load_pcg(args.data_name, args.split_name, args.train_percent)

    elif args.data_name.startswith("Hypersphere"):
        G = load_hyper_data(args.data_name, args.edge_name, args.split_name, args.train_percent)

    elif args.data_name == "yelp":
        G = import_yelp(args.data_name, args.split_name, args.train_percent)

    elif args.data_name == "Humloc":
        G = load_humloc()

    elif args.data_name == "Eukloc":
        G = load_eukloc()
    elif args.data_name == "ogbn-proteins":
        G = import_ogb(args.data_name)

    else:
        raise OSError("Dataset not found")

    if args.model_name == "LFLF_GCN":
        model = LFLF_GCN(in_channels=G.x.shape[1],
                         hidden_channels=args.hidden,
                         class_channels=G.y.shape[1],
                         num_layers=args.layer,
                         dropout=args.dropout,
                         )
    elif args.model_name == "LFLF_SAGE":
        model = LFLF_SAGE(in_channels=G.x.shape[1],
                          hidden_channels=args.hidden,
                          class_channels=G.y.shape[1],
                          num_layers=args.layer,
                          dropout=args.dropout,
                          )
    elif args.model_name == "LFLF_GAT":
        model = LFLF_GAT(in_channels=G.x.shape[1],
                         hidden_channels=args.hidden,
                         class_channels=G.y.shape[1],
                         num_layers=args.layer,
                         dropout=args.dropout,
                         )

    else:
        raise OSError("model not defined")

    train_loader = NeighborSampler(G.edge_index,
                                   sizes=args.sample_per_layer,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_nodes=G.num_nodes
                                   )

    val_loader = NeighborSampler(G.edge_index,
                                 sizes=args.sample_per_layer,
                                 node_idx=G.val_mask,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_nodes=G.num_nodes
                                 )

    # subgraph_loader = RawNeighborSampler(G.edge_index,
    #                                      sizes=[-1],
    #                                      batch_size=64,
    #                                      shuffle=False,
    #                                      num_nodes=G.num_nodes
    #                                      )
    print(G.edge_index.shape)
    subgraph_loader = NeighborLoader(G,
                                     input_nodes=None,
                                     num_neighbors=[-1],
                                     shuffle=False,
                                     batch_size=64)

    # subgraph_loader = RawNeighborSampler(G.edge_index,
    #                                      sizes=[-1, -1],
    #                                      input_nodes=G.test_mask,
    #                                      batch_size=32,
    #                                      shuffle=False,
    #                                      )
    print("loaders built")
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    model.to(device)
    x = G.x.to(device)
    labels = G.y.to(device)
    edge_index_full_graph = G.edge_index.to(device)
    edge_attr = G.edge_attr.to(device)
    soft_labels = G.soft_labels.to(device)
    train_mask = G.train_mask.to(device)
    val_mask = G.val_mask.to(device)
    test_mask = G.test_mask.to(device)

    for epoch in range(1, args.epochs):

        loss_train = model_train(train_loader)
        loss_val, model = model_val(val_loader)
        #micro_train, macro_train, micro_test, macro_test, micro_val, macro_val, roc_auc_train_macro, \
        #roc_auc_test_macro, roc_auc_val_macro, ap_test = batch_test(subgraph_loader)

        micro_train, macro_train, micro_test, macro_test, micro_val, macro_val, roc_auc_train_macro, \
        roc_auc_test_macro, roc_auc_val_macro, ap_test = batch_test(subgraph_loader)

        #micro_train, macro_train, micro_test, macro_test, roc_auc_train, roc_auc_test = model_test()

        print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
              f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f}, '
              f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f}, '
              #f'Train ROC-AUC micro: {roc_auc_train_micro:.4f}, '
              f'train ROC-AUC macro: {roc_auc_train_macro:.4f} '
              #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
              f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
              f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test Average Precision Score: {ap_test:.4f}, '
              )
        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Optimization Finished!")
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    micro_train, macro_train, micro_test, macro_test, micro_val, macro_val, roc_auc_train_macro, \
    roc_auc_test_macro, roc_auc_val_macro, ap_test = batch_test(subgraph_loader)

    #micro_train, macro_train, micro_test, macro_test, roc_auc_train, roc_auc_test = model_test()
    print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
          #f'Train ROC-AUC micro: {roc_auc_train_micro:.4f}, '
          f'train ROC-AUC macro: {roc_auc_train_macro:.4f} '
          #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
          f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
          #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
          f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
          f'Test Average Precision Score: {ap_test:.4f}, '
          )




