import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy
import time


class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self):
        pass


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0.,hop=1, name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.hop = hop
        self.a_list = torch.nn.ParameterList([nn.Parameter(torch.Tensor(self.dim, 1)) for i in range(hop)])

        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_list = []
        for i in range(self.hop):
            tmp = torch.matmul(a_input, self.a_list[i])
            tmp = self.leakyrelu(tmp).squeeze(-1).view(batch_size, N, N)
            e_list.append(tmp)


        mask = -9e15 * torch.ones_like(e_list[0])
        for i in range(self.hop):
            e_list[i] = torch.where(adj[:,i].eq(i+1), e_list[i], mask).exp()
            if i>1:
                e_list[i] = F.dropout(e_list[i], self.dropout, training=self.training)


        tmp = torch.stack(e_list).sum(dim=0)
        s = torch.sum(tmp, dim=-1, keepdim=True)
        s = torch.where(s.eq(0), torch.ones_like(s), s)
        alpha = tmp / s
        #0.0145
        output = torch.matmul(alpha, h)
        return output



class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        #self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None, t=1.0):
        if extra_vector is not None:
            batch_size = neighbor_vector.shape[0]
            neighbor_vector = neighbor_vector.view(batch_size, -1, self.dim)
            neighbor_weight = neighbor_weight.view(batch_size, -1)

            alpha = torch.matmul(extra_vector.unsqueeze(-2).repeat(1, neighbor_vector.shape[1], 1)*neighbor_vector, self.w_1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1) * t
            mask = -9e15 * torch.ones_like(alpha)
            alpha = torch.where(neighbor_weight==0, mask,alpha)
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2).unsqueeze(-2).repeat(1, self_vectors.shape[1], 1)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
        output = torch.cat([self_vectors, neighbor_vector, extra_vector.unsqueeze(-2).repeat(1, self_vectors.shape[1], 1)], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)

        output = self.act(output)
        return output

