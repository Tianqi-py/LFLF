#!/usr/bin/env python3
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv
from layers import Attention, Weighted_SAGE


class LFLF_SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = dropout

        self.mlp = torch.nn.Linear(hidden_channels, class_channels)
        self.attention = Attention(hidden_channels)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(Weighted_SAGE(in_channels, hidden_channels, aggr="mean"))

        self.label_conv = Weighted_SAGE(class_channels, hidden_channels, aggr="mean")

    def forward(self, x, y, adjs, edge_weights):

        for i, (edge_index, e_id, size) in enumerate(adjs):

            x_target = x[:size[1]]
            embedding_adj = self.convs[i]((x, x_target), edge_index)

            y_target = y[:size[1]]
            embedding_label_cor = self.label_conv((y, y_target), edge_index, edge_weights[i])

            embedding = torch.stack([embedding_adj, embedding_label_cor], dim=1)
            x, att = self.attention(embedding)

            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
                # construct new label matrix based on x
                y = self.sigmoid(self.mlp(x))
            else:
                y = self.sigmoid(self.mlp(x))

        return x, y

    @torch.no_grad()
    def full_forward(self, x, y, edge_index, edge_attr):

        for i, conv in enumerate(self.convs):

            x_adj = conv(x, edge_index)
            x_label_cor = self.label_conv(y, edge_index, edge_attr)

            embedding = torch.stack([x_adj, x_label_cor], dim=1)
            x, att = self.attention(embedding)

            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
                # construct new label matrix based on x
                y = self.sigmoid(self.mlp(x))
            else:
                y = self.sigmoid(self.mlp(x))

        return x, y

    @torch.no_grad()
    def inference(self, x_all, y_all, subgraph_loader):
        device = x_all.device
        for i, conv in enumerate(self.convs):
            xs = []
            ys = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                y = y_all[batch.n_id.to(y_all.device)].to(device)

                x_adj = conv(x, batch.edge_index.to(device))
                x_label_cor = self.label_conv(y, batch.edge_index.to(device), batch.edge_attr.to(device))

                embedding = torch.stack([x_adj, x_label_cor], dim=1)
                x, att = self.attention(embedding)
                if i != self.num_layers - 1:
                    x = x.relu()
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    # construct new label matrix based on x
                    y = self.sigmoid(self.mlp(x))
                xs.append(x[:batch.batch_size].cpu())
                ys.append(y[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
            y_all = torch.cat(ys, dim=0)
        return x_all, y_all


class LFLF_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = dropout

        self.attention = Attention(hidden_channels)
        self.mlp = nn.Linear(hidden_channels, class_channels)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_channels, hidden_channels))

        self.label_conv = GCNConv(class_channels, hidden_channels)

    def forward(self, x, y, adjs, edge_weights):

        for i, (edge_index, e_id, size) in enumerate(adjs):

            embedding_adj = self.convs[i](x, edge_index)[:size[1]]
            embedding_label_cor = self.label_conv(y, edge_index, edge_weights[i])[:size[1]]

            embedding = torch.stack([embedding_adj, embedding_label_cor], dim=1)
            x, att = self.attention(embedding)

            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
                # construct new label matrix based on x
                y = self.sigmoid(self.mlp(x))
            else:
                y = self.sigmoid(self.mlp(x))

        return x, y

    @torch.no_grad()
    def full_forward(self, x, y, edge_index, edge_attr):

        for i, conv in enumerate(self.convs):

            x_adj = conv(x, edge_index)
            x_label_cor = self.label_conv(y, edge_index, edge_attr)

            embedding = torch.stack([x_adj, x_label_cor], dim=1)
            x, att = self.attention(embedding)

            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
                # construct new label matrix based on x
                y = self.sigmoid(self.mlp(x))
            else:
                y = self.sigmoid(self.mlp(x))

        return x, y

    @torch.no_grad()
    def inference(self, x_all, y_all, subgraph_loader):
        device = x_all.device
        for i, conv in enumerate(self.convs):
            xs = []
            ys = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                y = y_all[batch.n_id.to(y_all.device)].to(device)
                x_adj = conv(x, batch.edge_index)
                x_label_cor = self.label_conv(y, batch.edge_index, batch.edge_attr.to(device))

                embedding = torch.stack([x_adj, x_label_cor], dim=1)
                x, att = self.attention(embedding)
                if i != self.num_layers - 1:
                    x = x.relu()
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    # construct new label matrix based on x
                    y = self.sigmoid(self.mlp(x))
                else:
                    y = self.sigmoid(self.mlp(x))
                xs.append(x[:batch.batch_size].cpu())
                ys.append(y[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
            y_all = torch.cat(ys, dim=0)
        return x_all, y_all


class LFLF_GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, class_channels, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = dropout
        self.mlp = torch.nn.Linear(hidden_channels, class_channels)

        self.attention = Attention(hidden_channels)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(GATConv(in_channels, hidden_channels, heads=2,
                                      concat=False, edge_dim=1))

        self.label_conv = GATConv(class_channels, hidden_channels, heads=2,
                                  concat=False, edge_dim=1)

    def forward(self, x, y, adjs, edge_weights):

        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]
            embedding_adj = self.convs[i]((x, x_target), edge_index)

            y_target = y[:size[1]]
            embedding_label_cor = self.label_conv((y, y_target), edge_index, edge_attr=edge_weights[i])

            embedding = torch.stack([embedding_adj, embedding_label_cor], dim=1)
            x, att = self.attention(embedding)

            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
                # construct new label matrix based on x
                y = self.sigmoid(self.mlp(x))
            else:
                y = self.sigmoid(self.mlp(x))

        return x, y

    @torch.no_grad()
    def full_forward(self, x, y, edge_index, edge_attr):

        for i, conv in enumerate(self.convs):

            x_adj = conv(x, edge_index)
            x_label_cor = self.label_conv(y, edge_index, edge_attr)

            embedding = torch.stack([x_adj, x_label_cor], dim=1)
            x, att = self.attention(embedding)

            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
                # construct new label matrix based on x
                y = self.sigmoid(self.mlp(x))
            else:
                y = self.sigmoid(self.mlp(x))

        return x, y