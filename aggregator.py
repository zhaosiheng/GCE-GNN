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
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        #self.w_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_sc = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.a_sc = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))
        self.leakyrelu = nn.LeakyReLU(0.2)


        self.q_list = torch.nn.ParameterList([nn.Parameter(torch.Tensor(self.dim + 1, 1)) for i in range(6)])

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            #alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            #alpha = torch.matmul(extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, self.w_1).squeeze(-1)

            e = self_vectors.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1) * neighbor_vector
            e = torch.cat([e, neighbor_weight.unsqueeze(-1)], -1)

            t0 = time.time()
            e_list = []
            for i in range(6):
                tmp = self.leakyrelu(torch.matmul(e, self.q_list[i])).squeeze(-1)
                e_list.append(tmp)


            h0_list = []
            a_list = []
            mask = -9e15 * torch.ones_like(e_list[0])
            for i in range(6):
                alpha = torch.where(torch.logical_and(neighbor_weight >= pow(4,i), neighbor_weight < pow(4,i+1)), e_list[i], mask)
                alpha = torch.softmax(alpha, -1)
                padding = alpha[0][0][0]
                zero = torch.zeros_like(alpha)
                alpha = torch.where(alpha == padding, zero, alpha)
                h0 = (alpha.unsqueeze(-1) * neighbor_vector).sum(-2)
                h0_list.append(h0)

            h0 = torch.stack(h0_list, -2)

            '''
            h1_list = []
            for i in range(6):
                beta = torch.nn.functional.linear(h0_list[i], self.w_sc, self.bias)
                
                zero = torch.zeros_like(beta)
                beta = torch.where(h0_list[i] == 0, zero, beta)
                
                beta = torch.tanh(beta)
                beta = torch.matmul(beta, self.a_sc)
                padding = h0_list[0][0][0][0]
                zero = torch.zeros_like(beta)
                beta = torch.where(h0_list[i].sum(-1).unsqueeze(-1) == padding, zero, beta)
                h1_list.append(beta)
            '''



            '''
            for i in range(6):
                alpha = torch.where(torch.logical_and(neighbor_weight >= pow(4,i), neighbor_weight < pow(4,i+1)), e_list[i], mask)
                a_list.append(alpha)
            alpha = torch.stack(a_list, -2)
            alpha = torch.softmax(alpha, -1)
            padding = alpha[0][0]
            zero = torch.zeros_like(alpha)
            alpha = torch.where(alpha == padding, zero, alpha)
            tmp = (alpha.unsqueeze(-1) * neighbor_vector.unsqueeze(-3).repeat(1,1,6,1,1)).sum(-2)
            '''


            beta = torch.nn.functional.linear(h0, self.w_sc, self.bias)
            beta = torch.tanh(beta)
            beta = torch.matmul(beta, self.a_sc)
            mask = -9e15 * torch.ones_like(beta)
            beta = torch.where(h0.sum(-1).unsqueeze(-1) == 0, mask, beta).squeeze(-1)


            beta = torch.softmax(beta, -1)
            neighbor_vector = (beta.unsqueeze(-1) * h0).sum(-2)
            t4 = time.time()
            r4 = t4 - t0

        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
        output = torch.cat([self_vectors, neighbor_vector], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        return output
