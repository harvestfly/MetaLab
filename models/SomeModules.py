from some_libs import *
import torch
import torch.nn as nn
from torch.nn.init import *
from torch.nn import functional as F

def emb_init(x):
    x = x.weight.data
    sc = 2 / (x.size(1) + 1)
    x.uniform_(-sc, sc)

# https://github.com/kuangliu/pytorch-groupnorm/blob/master/groupnorm.py
class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        #N,C,H,W = x.size()
        size = x.size()
        N, C = x.size()
        H,W=1,1
        G = self.num_groups
        assert C % G == 0
        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x-mean) / (var+self.eps).sqrt()
        #x = x.view(N,C,H,W)
        x = x.view(N, C)
        return x * self.weight + self.bias

# https://discuss.pytorch.org/t/adaptive-normalization/9157/2
class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)

class AdaptiveBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.ones(1, 1))
        self.b = nn.Parameter(torch.ones(1, 1))
        #self.b = nn.Parameter(torch.ones(1, 1) )

        # x = self.a.data;    x.uniform_(-1, 1)

    def forward(self, x):
        #print("a={} b={}",self.a.data[0],self.b.data[0])
        return self.a * x + self.b * self.bn(x)

class Embedding_NN(nn.Module):
    loss_curve=[]

    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops, use_bn="none", classify=False,
                 isDrop=False):
        init_weights = nn.init.kaiming_normal_  # kaiming_normal_
        super().__init__()  ## inherit from nn.Module parent class
        if emb_szs is None:
            self.embs, n_emb = None, 0
        else:
            self.embs = nn.ModuleList([nn.Embedding(m, d) for m, d in emb_szs])  ## construct embeddings
            for emb in self.embs:
                emb_init(emb)  ## initialize embedding weights
            n_emb = sum(e.embedding_dim for e in self.embs)  ## get embedding dimension needed for 1st layer
            szs = [n_emb + n_cont] + szs  ## add input layer to szs
        print("******Embedding_NN n_cont={} n_emb={} szs={}\n\t use_bn={} classify={}\n******Embedding_NN\n".
              format(n_cont, n_emb, szs, use_bn, classify))
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i + 1]) for i in
            range(len(szs) - 1)])  ## create linear layers input, l1 -> l1, l2 ...

        self.input_bn = None    #输入为curve,  bn不合理
        if use_bn=="bn": ## batchnormalization for hidden layers activations
            self.bns = nn.ModuleList([nn.BatchNorm1d(sz) for sz in szs[1:]])
            self.input_bn = nn.BatchNorm1d(n_cont)
        elif use_bn == "adaptive":  ## batchnormalization for hidden layers activations
            self.bns = nn.ModuleList([AdaptiveBatchNorm1d(sz) for sz in szs[1:]])
            self.input_bn = AdaptiveBatchNorm1d(n_cont)
        else:
            self.bns = [None]*len(self.lins)

        for o in self.lins:
            init_weights(o.weight.data)  ## init weights with kaiming normalization
        self.outp = nn.Linear(szs[-1], out_sz)  ## create linear from last hidden layer to output
        init_weights(self.outp.weight.data)  ## do kaiming initialization
        if isDrop:
            #self.emb_drop = nn.Dropout(emb_drop)  ## embedding dropout, will zero out weights of embeddings
            #self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])  ## fc layer dropout
            self.drops = nn.ModuleList([None if drop==0 else nn.Dropout(drop) for drop in drops])
        else:
            self.drops = [None]*len(self.lins)


        self.classify = classify
        self.isDrop = isDrop
        # print(TorchSummarizeDf(self).make_df())
        print(self)


    def forward(self, x_cat, x_cont):
        if self.input_bn is None:## apply batchnorm to continous variables
            x2 = x_cont
        else:
            x2 = self.input_bn(x_cont)
        if x_cat is not None:
            x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]  # takes necessary emb vectors
            x = torch.cat(x, 1)  ## concatenate along axis = 1 (columns - side by side) # this is our input from cats
            if self.isDrop:
                x = self.emb_drop(x)  ## apply dropout to elements of embedding tensor
                x = torch.cat([x, x2], 1)  ## concatenate cats and conts for final input
        else:
            x = x2

        for l, d, b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            #x = F.selu(l(x))
            #x = F.elu(l(x))
            #x = l(x)
            if b is not None:
                x = b(x)  ## apply batchnorm activations
            if d is not None:
                x = d(x)  ## apply dropout to activations

        x = self.outp(x)  # we defined this externally just not to apply dropout to output
        # x = F.log_softmax(x)
        if self.classify:
            x = F.sigmoid(x)  # for classification
        return x