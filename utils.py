import os
import torch
import scipy
import scipy.io
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, sort_edge_index
from torch_geometric.datasets import Yelp
from torch_geometric.data import DataLoader
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_tensor(a):
    """Row-normalize tensor that requires grad"""
    rowsum = a.sum(1).detach().cpu().numpy()
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(a.detach().cpu().numpy())
    mx = torch.tensor(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# blogcatalog, flickr and youtube
def load_mat_data(data_name, split_name, train_percent, path="data/"):
    print('Loading dataset ' + data_name + '.mat...')
    mat = scipy.io.loadmat(path + data_name)

    labels = mat['group']
    labels = sparse_mx_to_torch_sparse_tensor(labels).to_dense()

    adj = mat['network']
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
    edge_index = torch.transpose(torch.nonzero(adj), 0, 1)

    # node_index = torch.arange(labels.shape[0])
    # self_loop = torch.vstack((node_index, node_index))
    # edge_index = torch.cat((edge_index, self_loop), 1)
    edge_weight = torch.sum(labels[edge_index[0, :]] * labels[edge_index[1, :]], 1).float()

    # prepare the feature matrix
    #features = torch.range(0, labels.shape[0] - 1).long()
    features = torch.eye(labels.shape[0])
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

    y = labels.clone().detach()
    y[val_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])
    y[test_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])

    #num_class = labels.shape[1]
    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             edge_attr=edge_weight,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.soft_labels = y
    G.n_id = torch.arange(num_nodes)

    #return adj, edge_index, features, labels, y, train_mask, val_mask, test_mask, num_class, num_nodes
    return G


def load_pcg(data_name, split_name, train_percent, path="../data/"):
    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()
    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()

    edges = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edges_undir.csv"),
                                       dtype=np.dtype(float), delimiter=','))
    edge_index = torch.transpose(edges, 0, 1).long()
    edge_weight = torch.sum(labels[edge_index[0, :]] * labels[edge_index[1, :]], 1).float()

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
    #print(test_mask[:5])

    y = labels.clone().detach().float()
    y[val_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])
    y[test_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])

    #num_class = labels.shape[1]
    #num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             edge_attr=edge_weight,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.soft_labels = y

    #return edge_weight, edge_index, features, labels, y, train_mask, val_mask, test_mask, num_class, num_nodes
    return G


def load_hyper_data(data_name, edge_name, split_name, train_percent, path="../../../data/"):
    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           skip_header=1, dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            skip_header=1, dtype=np.dtype(float), delimiter=',')).float()

    edges = torch.tensor(np.genfromtxt(os.path.join("../../", edge_name),
                         dtype=np.dtype(float), delimiter=',')).long()
    edge_index = torch.transpose(edges, 0, 1)

    edge_weight = torch.sum(labels[edge_index[0, :]] * labels[edge_index[1, :]], 1).float()
    #edge_weight = torch.ones(edge_index.shape[])
    #label_cor = torch.sparse_coo_tensor(edge_index, label_cor, [labels.shape[0], labels.shape[0]])

    #edge_weight = normalize_tensor(label_cor.to_dense().fill_diagonal_(1.0))
    #edge_weight = edge_weight[edge_index[0, :], edge_index[1:, ]][0]

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
             edge_attr=edge_weight,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.soft_labels = y
    G.n_id = torch.arange(num_nodes)

    #return adj, edge_index, features, labels, y, train_mask, val_mask, test_mask, num_class, num_nodes #, iso_mask
    return G


def import_yelp(data_name, split_name, train_percent, path="../data/"):
    print('Loading dataset ' + data_name + '...')
    dataset = Yelp(root='../tmp/Yelp')
    data = dataset[0]
    labels = data.y
    features = data.x
    edge_index = data.edge_index
    edge_weight = torch.sum(labels[edge_index[0, :]] * labels[edge_index[1, :]], 1).float()

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

    G = Data(x=features,
             edge_index=edge_index,
             edge_attr=edge_weight,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.soft_labels = y

    return G


def import_ogb(data_name):
    print('Loading dataset ' + data_name + '...')

    dataset = PygNodePropPredDataset(name=data_name, transform=T.ToSparseTensor(attr='edge_attr'))
    data = dataset[0]
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    row, col, _ = data.adj_t.coo()
    edge_weight = torch.sum(data.y[row] * data.y[col], 1).float()
    edge_index = torch.vstack((row, col))

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    val_mask[valid_idx] = True

    test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    y = data.y.clone().detach().float()
    y[valid_idx] = torch.full((1, data.y.shape[1]), 1 / data.y.shape[1])
    y[test_idx] = torch.full((1, data.y.shape[1]), 1 / data.y.shape[1])

    num_nodes = data.x.shape[0]

    G = Data(x=data.x,
             edge_index=edge_index,
             edge_attr=edge_weight,
             y=data.y)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.soft_labels = y
    G.n_id = torch.arange(num_nodes)

    return G


def load_humloc(data_name="HumanGo", path="data/"):
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edge_list.csv"),
                                           skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    edge_weight = torch.sum(labels[edge_index[0, :]] * labels[edge_index[1, :]], 1).float()

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()

    file_path = os.path.join(path, data_name, "split.pt")
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

    G = Data(x=features,
             edge_index=edge_index,
             edge_attr=edge_weight,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.soft_labels = y

    return G


def load_eukloc(data_name="EukaryoteGO", path="data/"):
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edge_list.csv"),
                                           skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    edge_weight = torch.sum(labels[edge_index[0, :]] * labels[edge_index[1, :]], 1).float()

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()

    file_path = os.path.join(path, data_name, "split.pt")
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

    G = Data(x=features,
             edge_index=edge_index,
             edge_attr=edge_weight,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.soft_labels = y

    return G


