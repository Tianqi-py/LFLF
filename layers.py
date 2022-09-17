#!/usr/bin/env python3
# coding: utf-8
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from typing import Tuple, Union
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch import Tensor
from torch_sparse import SparseTensor, matmul


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # unsoftmaxed attention vector
        w = self.project(z)
        # softmax attention vector
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class Weighted_SAGE(MessagePassing):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 aggr: str = 'mean',
                 bias: bool = True,
                 **kwargs,
                 ):
        super(Weighted_SAGE, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = nn.Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)