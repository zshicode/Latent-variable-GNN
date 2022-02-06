import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import dgl.function as fn
import networkx as nx
import time
# from dgl.nn.pytorch.conv import GraphConv,SAGEConv,GATConv
from sklearn.metrics import f1_score
from sklearn.preprocessing import minmax_scale
import scipy.sparse as sp

torch.manual_seed(72)

def normalized(wmat):
    deginvsqrt = torch.diag(torch.pow(wmat.sum(1),-0.5))
    deginvsqrt[torch.isnan(deginvsqrt)] = 0.0
    W = deginvsqrt.mm(wmat).mm(deginvsqrt)
    return W

def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.BoolTensor(data.train_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro

class GraphConv(nn.Module):
    # my implementation of GCN
    def __init__(self,in_dim,out_dim,drop=0.0,bias=False,activation=None):
        super(GraphConv,self).__init__()
        self.dropout = nn.Dropout(drop)
        #self.w = nn.Parameter(torch.randn(in_dim, out_dim))
        self.activation = activation
        self.w = nn.Linear(in_dim,out_dim,bias=bias)
        nn.init.xavier_uniform_(self.w.weight)
        # self.bias = bias
        # if self.bias:
        #     self.b = nn.Parameter(torch.zeros(1, out_dim))
        
    '''
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_uniform_(self.w)
        if self.bias:
            nn.init.zeros_(self.b)
    '''
    
    def forward(self,adj,x):
        x = self.dropout(x)
        x = adj.mm(x)
        x = self.w(x)
        '''
        x = x.mm(self.w)
        if self.bias:
            x += self.b
        '''
        if self.activation:
            return self.activation(x)
        else:
            return x

g, features, labels, mask = load_cora_data()
adjd = g.adjacency_matrix().to_dense()
lap = torch.eye(features.size()[0]) - normalized(adjd)
case = 12

if case == 1:
    adj = torch.eye(features.size()[0]) - 2*lap # 0.4490
elif case == 2:
    adj = torch.eye(features.size()[0]) - 0.5*lap # 0.7850
elif case == 3:
    adj = torch.exp(-lap) # use exp(-lap), exp(-2*lap), or exp(-0.5*lap), are all 0.1120
elif case == 4:
    adj = torch.eye(features.size()[0]) - torch.pow(lap,2) # 0.2380
elif case == 5:
    adj = torch.inverse(torch.eye(features.size()[0]) + lap) # 0.8120
elif case == 6:
    adj = torch.eye(features.size()[0]) - 0.66*lap # 0.8070
elif case == 7:
    adj = torch.inverse(torch.eye(features.size()[0]) + 2*lap) # 0.8370
elif case == 8:
    adj = torch.inverse(torch.eye(features.size()[0]) + 0.5*lap) # 0.7570
elif case == 9:
    adj = torch.inverse(torch.eye(features.size()[0]) + 0.66*lap) # 0.7840
elif case == 10:
    adj = torch.inverse(torch.eye(features.size()[0]) + 5*lap) # 0.8340
elif case == 11:
    adj = torch.pow(torch.eye(features.size()[0]) - lap, 2) # 0.3100
    # adj = torch.pow(torch.eye(features.size()[0]) - lap, 3) # 0.3100
elif case == 12:
    adj = torch.inverse(torch.eye(features.size()[0]) + 1.5*lap) # 0.7570
else:
    adj = torch.eye(features.size()[0]) - lap # 0.8220

class GCN(nn.Module):
    def __init__(self,g,in_dim,hidden_dim,out_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(in_dim,hidden_dim,bias=False,activation=F.relu)
        self.gc2 = GraphConv(hidden_dim,out_dim,bias=False,activation=None)
        self.dropout = nn.Dropout(0.5)
        self.g = g
    
    def forward(self,x):
        g = self.g.local_var()
        x = self.dropout(x)
        x = self.gc1(adj,x)
        x = self.gc2(adj,x)
        return x

idx_train = torch.LongTensor(range(140))
idx_val = torch.LongTensor(range(2208,2708))
idx_test = torch.LongTensor(range(140,1140))
net = GCN(g,in_dim=features.size()[1],hidden_dim=16,out_dim=labels.max()+1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=5e-4)

def evaluate(feat,idx):
    with torch.no_grad():
        net.eval()
        logits = net(feat)
        loss = F.cross_entropy(logits[idx],labels[idx])
        acc = accuracy(logits[idx],labels[idx])
    
    return acc,loss.detach().numpy()

# main loop
dur = []
losses = []
for epoch in range(50):
    if epoch >= 5:
        t0 = time.time()
        # if losses[-1]>np.mean(losses[-6:-1]):
        #     print('early stop at epoch {}'.format(epoch))
        #     break
    
    logits = net(features)
    loss = F.cross_entropy(logits[idx_train],labels[idx_train])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 5:
        dur.append(time.time() - t0)
    
    acc_val,val_loss = evaluate(features,idx_val)
    losses.append(val_loss)
    print("Epoch {} | Loss {:.4f} | Time(s) {:.4f} | Acc_Val {:.4f} | Loss_Val {:.4f}".format(
        epoch, loss.item(), np.mean(dur),acc_val,val_loss))

acc_test,_ = evaluate(features,idx_test)
print('Acc_Test {:.4f}'.format(acc_test))