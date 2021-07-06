import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F

class LineConv(Module):
    def __init__(self, layers=3,batch_size=100,emb_size=100):
        super(LineConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers
    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = trans_to_cuda(torch.FloatTensor(1,self.emb_size).fill_(0))
        # zeros = torch.zeros([1,self.emb_size])
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len)
        session = [session_emb_lgcn]
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)
        session_emb_lgcn = np.sum(session, 0)
        return session_emb_lgcn
    
class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=opt.long_edge_dropout, hop=opt.hop)
        self.LineGraph = LineConv(batch_size=self.batch_size, emb_size=self.dim)
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, mask, inputs):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        session_emb = self.sessiongraph(inputs, mask)
        con_loss = SSL(select, session_emb)
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores, con_loss

    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        h_local = self.local_agg(h, adj, mask_item)

        # global
        
        # combine
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        
        output = h_local 

        return output
    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def sessiongraph(self, sessions, mask):
        A, D = self.get_overlap(sessions)
        A = trans_to_cuda(torch.Tensor(A))
        D = trans_to_cuda(torch.Tensor(D))
        session_emb = self.LineGraph(self.embedding.weight, D, A, sessions, mask.squeeze(-1).sum(-1,keepdim=True))
        return session_emb

def SSL(sess_emb_hgnn, sess_emb_lgcn):
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding

    def row_column_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
        return corrupted_embedding

    def score(x1, x2):
        return torch.sum(torch.mul(x1, x2), 1)

    pos = score(sess_emb_hgnn, sess_emb_lgcn)
    neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
    one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
    # one = zeros = torch.ones(neg1.shape[0])
    con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
    return con_loss

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()

    hidden = model(items, adj, mask, inputs)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    scores, con_loss = model.compute_scores(seq_hidden, mask, inputs)
    return targets, scores, con_loss


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores, con_loss = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1) + con_loss * model.opt.lamda
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr, hit_alias, mrr_alias = [], [], [], []
    for data in test_loader:
        targets, scores, con_loss = forward(model, data)
        sub_scores = scores.topk(20)[1]
        sub_scores_alias = scores.topk(10)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        sub_scores_alias = trans_to_cpu(sub_scores_alias).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            #@20
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
                
        for score, target, mask in zip(sub_scores_alias, targets, test_data.mask):
            #@10
            hit_alias.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_alias.append(0)
            else:
                mrr_alias.append(1 / (np.where(score == target - 1)[0][0] + 1))
            

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)
    
    result.append(np.mean(hit_alias) * 100)
    result.append(np.mean(mrr_alias) * 100)
    return result
