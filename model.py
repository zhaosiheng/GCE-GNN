import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.plugins.training_type import TPUSpawnPlugin
from utils import *


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
        self.adj_all = torch.Tensor(adj_all).long()
        self.num = torch.Tensor(num).float()
        self.degree = self.num.sum(-1)
        self.epoch = 0
        

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=opt.long_edge_dropout, hop=opt.hop)
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

    def compute_scores(self, hidden, mask, s_global):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1) +s_global
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores

    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        h_local = self.local_agg(h, adj, mask_item)



        # combine
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        s_global = 0
        output = h_local 

        return output, s_global


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

    hidden, s_global = model(items, adj, mask, inputs)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask, s_global)


def train_test(model, train_data, test_data, opt):

    dm = Litdatamodule(model.model.batch_size, train_data, test_data, opt)
    trainer = Trainer(max_epochs=1, tpu_cores=8)
    trainer.fit(model, dm)
    #trainer.test(model, dm)
    
class Litdatamodule(LightningDataModule):
    def __init__(self, batch_size ,train, test, opt):
        super().__init__()
        self.batch_size = batch_size
        self.tmp1 = train
        self.tmp2 =test
        self.opt = opt
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = Data(self.tmp1, hop=self.opt.hop)
        if stage == 'test' or stage is None:
            self.test_data = Data(self.tmp2, hop=self.opt.hop)
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, num_workers=4, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, num_workers=4, batch_size=self.batch_size, shuffle=False)
class Litmodel(LightningModule):
    def __init__(self,opt, num_node, adj, num):
        super().__init__()
        self.save_hyperparameters()
        self.model = CombineGraph(opt, num_node, adj, num)
        self.best_result = [0, 0, 0, 0]
        self.best_epoch = [0, 0, 0, 0]
        self.opt = opt
        
    def forward(self, data):
        alias_inputs, adj, items, mask, targets, inputs = data
        alias_inputs = alias_inputs.long()
        items = items.long()
        adj = adj.float()
        mask = mask.long()
        inputs = inputs.long()

        hidden, s_global = self.model(items, adj, mask, inputs)
        get = lambda index: hidden[index][alias_inputs[index]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        return targets, self.model.compute_scores(seq_hidden, mask, s_global)
    def training_step(self, batch, batch_idx):
        data = batch
        targets, scores = self(data)
        targets = targets.long()
        loss = self.model.loss_function(scores, targets - 1) 
        self.log('train_loss', loss)
        return loss
    def configure_optimizers(self):
        return self.model.optimizer
'''
def test_step(self, batch, batch_idx):
        data = batch
        targets, scores = self(data)

        result = []
        hit, mrr, hit_alias, mrr_alias = [], [], [], []
        for data in test_loader:
            targets, scores = self( data)
            sub_scores = scores.topk(20)[1]
            sub_scores_alias = scores.topk(10)[1]
            sub_scores = sub_scores.detach().numpy()
            sub_scores_alias = sub_scores_alias.detach().numpy()
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
    def test_epoch_end(self, result):
        hit, mrr, hit_alias, mrr_alias = reslut
        flag = 0
        if hit >= best_result[0]:
            self.best_result[0] = hit
            self.best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            self.best_result[1] = mrr
            self.best_epoch[1] = epoch
            flag = 1
        if hit_alias >= best_result[2]:
            self.best_result[2] = hit_alias
            self.best_epoch[2] = epoch
            flag = 1
        if mrr_alias >= best_result[3]:
            self.best_result[3] = mrr_alias
            self.best_epoch[3] = epoch
            flag = 1
        self.print('Current Result:')
        self.print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tRecall@10:\t%.4f\tMMR@10:\t%.4f' % (hit, mrr, hit_alias, mrr_alias))
        self.print('Best Result:')
        self.print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tEpoch:\t%d,\t%d,\t%d,\t%d' % (
            best_result[0], best_result[1], best_result[2], best_result[3], best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3]))
'''

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch_xla.distributed.parallel_loader as pl
import time

def map_fn(index, flags):
  ## Setup 

  # Sets a common random seed - both for initialization and ensuring graph is the same
  torch.manual_seed(flags['seed'])

  # Acquires the (unique) Cloud TPU core corresponding to this process's index
  device = xm.xla_device()  


  ## Dataloader construction

  # Creates the transform for the raw Torchvision data
  # See https://pytorch.org/docs/stable/torchvision/models.html for normalization
  # Pre-trained TorchVision models expect RGB (3 x H x W) images
  # H and W should be >= 224
  # Loaded into [0, 1] and normalized as follows:
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
  to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
  resize = transforms.Resize((224, 224))
  my_transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

  # Downloads train and test datasets
  # Note: master goes first and downloads the dataset only once (xm.rendezvous)
  #   all the other workers wait for the master to be done downloading.

  if not xm.is_master_ordinal():
    xm.rendezvous('download_only_once')

  train_dataset = datasets.FashionMNIST(
    "/tmp/fashionmnist",
    train=True,
    download=True,
    transform=my_transform)

  test_dataset = datasets.FashionMNIST(
    "/tmp/fashionmnist",
    train=False,
    download=True,
    transform=my_transform)
  
  if xm.is_master_ordinal():
    xm.rendezvous('download_only_once')
  
  # Creates the (distributed) train sampler, which let this process only access
  # its portion of the training dataset.
  train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=True)
  
  test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=False)
  
  # Creates dataloaders, which load data in batches
  # Note: test loader is not shuffled or sampled
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=flags['batch_size'],
      sampler=train_sampler,
      num_workers=flags['num_workers'],
      drop_last=True)

  test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=flags['batch_size'],
      sampler=test_sampler,
      shuffle=False,
      num_workers=flags['num_workers'],
      drop_last=True)
  

  ## Network, optimizer, and loss function creation

  # Creates AlexNet for 10 classes
  # Note: each process has its own identical copy of the model
  #  Even though each model is created independently, they're also
  #  created in the same way.
  net = torchvision.models.alexnet(num_classes=10).to(device).train()

  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters())


  ## Trains
  train_start = time.time()
  for epoch in range(flags['num_epochs']):
    para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
    for batch_num, batch in enumerate(para_train_loader):
      data, targets = batch 

      # Acquires the network's best guesses at each class
      output = net(data)

      # Computes loss
      loss = loss_fn(output, targets)

      # Updates model
      optimizer.zero_grad()
      loss.backward()

      # Note: optimizer_step uses the implicit Cloud TPU context to
      #  coordinate and synchronize gradient updates across processes.
      #  This means that each process's network has the same weights after
      #  this is called.
      # Warning: this coordination requires the actions performed in each 
      #  process are the same. In more technical terms, the graph that
      #  PyTorch/XLA generates must be the same across processes. 
      xm.optimizer_step(optimizer)  # Note: barrier=True not needed when using ParallelLoader 

  elapsed_train_time = time.time() - train_start
  print("Process", index, "finished training. Train time was:", elapsed_train_time) 


  ## Evaluation
  # Sets net to eval and no grad context 
  net.eval()
  eval_start = time.time()
  with torch.no_grad():
    num_correct = 0
    total_guesses = 0

    para_train_loader = pl.ParallelLoader(test_loader, [device]).per_device_loader(device)
    for batch_num, batch in enumerate(para_train_loader):
      data, targets = batch

      # Acquires the network's best guesses at each class
      output = net(data)
      best_guesses = torch.argmax(output, 1)

      # Updates running statistics
      num_correct += torch.eq(targets, best_guesses).sum().item()
      total_guesses += flags['batch_size']
  
  elapsed_eval_time = time.time() - eval_start
  print("Process", index, "finished evaluation. Evaluation time was:", elapsed_eval_time)
  print("Process", index, "guessed", num_correct, "of", total_guesses, "correctly for", num_correct/total_guesses * 100, "% accuracy.")
