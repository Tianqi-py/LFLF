import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, class_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.sigmoid(x)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, class_channels):
        super().__init__()

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.5)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, class_channels, heads=1, concat=False,
                             dropout=0.5)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.sigmoid(x)


class SAGE_sup(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, class_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return F.sigmoid(x)

    # @torch.no_grad()
    # def full_forward(self, x, adjs):
    #     for i, (edge_index, _, size) in enumerate(adjs):
    #         x_target = x[:size[1]]  # Target nodes are always placed first.
    #         x = self.convs[i]((x, x_target), edge_index)
    #         if i != self.num_layers - 1:
    #             x = x.relu()
    #             x = F.dropout(x, p=0.5, training=self.training)
    #
    #     return F.sigmoid(x)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id]
                x = conv(x, batch.edge_index)
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
        return F.sigmoid(x_all)

    # @torch.no_grad()
    # def full_forward(self, x, edge_index):
    #     for i, conv in enumerate(self.convs):
    #         x = conv(x, edge_index)
    #         if i != self.num_layers - 1:
    #             x = x.relu()
    #             x = F.dropout(x, p=0.5, training=self.training)
    #         else:
    #             x = F.sigmoid(x)
    #     return x


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, class_channels)

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        out = self.fc3(x)
        #out=x
        return F.sigmoid(out)


class H2GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(H2GCN, self).__init__()
        # input
        self.dense1 = torch.nn.Linear(nfeat, nhid)
        # output
        self.dense2 = torch.nn.Linear(nhid*7, nclass)
        # drpout
        # self.dropout = SparseDropout(dropout)
        self.dropout = dropout
        # conv
        self.conv1 = GCNConv(nhid, nhid,
                             #cached=True, normalize=False
                             )
        self.conv2 = GCNConv(nhid*2, nhid*2,
                             #cached=True, normalize=False
                             )
        self.relu = torch.nn.ReLU()
        self.vec = torch.nn.Flatten()
        self.iden = torch.sparse.Tensor()

    def forward(self, features, edge_index):

        # feature space ----> hidden
        # adj2 = adj * adj
        # r1: compressed feature matrix
        x = self.relu(self.dense1(features))
        # # vectorize
        # x = self.vec(x)
        # aggregate info from 1 hop away neighbor
        # r2 torch.cat(x, self.conv(x, adj), self.conv(x, adj2))
        x11 = self.conv1(x, edge_index)
        x12 = self.conv1(x11, edge_index)
        x1 = torch.cat((x11, x12), -1)

        # vectorize
        # x = self.vec(x1)
        # aggregate info from 2 hp away neighbor
        x21 = self.conv2(x1, edge_index)
        x22 = self.conv2(x21, edge_index)
        x2 = torch.cat((x21, x22), -1)

        # concat
        x = torch.cat((x, x1, x2), dim=-1)
        # x = self.dropout(x)
        x = F.dropout(x, self.dropout)
        x = self.dense2(x)

        return F.sigmoid(x)
