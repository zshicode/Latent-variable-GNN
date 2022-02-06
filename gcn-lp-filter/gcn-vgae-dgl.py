import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cvxopt import matrix,solvers
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import dgl.function as fn
import networkx as nx
import time
from dgl.nn.pytorch.conv import GraphConv,SAGEConv,GATConv
from sklearn.metrics import f1_score
from sklearn.preprocessing import minmax_scale
solvers.options['show_progress'] = False
import scipy.sparse as sp

model = 'VGAE'
torch.manual_seed(72)

def neighborhood(feat,k):
    #计算距离矩阵dmat和邻接矩阵C
    featprod = np.dot(feat.T,feat)
    smat = np.tile(np.diag(featprod),(feat.shape[1],1))
    dmat = smat + smat.T - 2*featprod
    dsort = np.argsort(dmat)[:,1:k+1]
    C = np.zeros((feat.shape[1],feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i,j] = 1.0
    
    return dmat,C

def normalized(wmat):
    degpow = np.diag(np.power(
        np.sum(wmat,axis=0),-0.5))
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow,wmat),degpow)

    return W

def optimize_w(i,k,feat,dsort):
    dist = []
    for j in dsort[i]:
        dist.append(feat.T[i]-feat.T[j])
    
    dist = np.array(dist)
    gram = np.dot(dist,dist.T)
    w = np.zeros(feat.shape[1])
    mu = 2
    # gram = minmax_scale(gram,axis=1)
    # ginv = np.linalg.inv(gram+0.01*np.eye(gram.shape[0]))
    # ww = ginv.sum(axis=0)/ginv.sum()
    gram = gram + mu*np.eye(gram.shape[0])
    Q = 2 * matrix(gram)
    p = matrix(np.zeros(k))  # 代表一次项的系数
    G = -1 * matrix(np.eye(k))  # G和h代表GX+s = h，s>=0,表示每一个变量x均大于零
    h = p
    A = matrix(np.ones(k),(1,k))
    b = matrix(1.0)                                                    # AX = b
    sol = solvers.qp(Q, p, G, h, A, b)
    '''
    solw = np.ones(k)/k
    e = np.ones(k)
    for _ in range(50):
        solw = mu*solw/(np.dot(gram,solw)+mu*np.dot(np.dot(e,e.T),solw))

    '''
    for j in range(k):
        w[dsort[i,j]] = sol['x'][j]
        #w[dsort[i,j]] = solw[j]
        #w[dsort[i,j]] = ww[j]
    
    return w

def buildgraph(feat,method='lp',sigma=0.01,k=5):
    feat = minmax_scale(feat,axis=1)
    featprod = np.dot(feat.T,feat)
    dmat,C = neighborhood(feat,k)
    wmat = np.exp(-dmat/(2*sigma**2))
    W = normalized(wmat)
    if method == 'lnpbasic':
        w = []
        dsort = np.argsort(dmat)[:,1:k+1]
        for i in range(feat.shape[1]):
            w.append(optimize_w(i,k,feat,dsort))
        
        W = np.array(w)
    else:
        e = np.ones(feat.shape[1])
        mu = 2
        iteration = 100
        W = np.ones((feat.shape[1],feat.shape[1]))/(feat.shape[1]-1) - np.eye(feat.shape[1])
        for i in range(iteration):
            W = W * (featprod+mu*np.dot(e,e.T)) / (C*W*featprod+mu*C*W*np.dot(e,e.T))
        
        W[np.isnan(W)] = 0
        W = W/k
    
    return W

def label_prop(feat,label,nclass,alpha=0.99,iteration=100):
    y = torch.zeros(feat.size(0),nclass)
    for i in range(len(label)):
        y[i,label[i]] = 1.0
    
    '''
    featsum = torch.diag(torch.pow(feat.sum(dim=1),-1))
    featsum[torch.isnan(featsum)] = 0.0
    feat = featsum.mm(feat)
    '''
    # featprod = torch.mm(feat,feat.t())
    # smat = torch.diag(featprod).expand(1,feat.size(0))
    # dmat = smat + smat.t() - 2*featprod
    # sigma = 0.1
    # #k = 5
    # wmat = torch.exp(-dmat/(2*sigma**2))
    # deginvsqrt = torch.diag(torch.pow(wmat.sum(1),-0.5))
    # deginvsqrt[torch.isnan(deginvsqrt)] = 0.0
    # W = deginvsqrt.mm(wmat).mm(deginvsqrt)
    # W = W.detach().numpy()
    # f = y
    # for _ in range(iteration):
    #     f = alpha*W.mm(f) + (1-alpha)*y
    f0 = y.detach().numpy()
    f = f0
    feat = feat.detach().numpy()
    W = buildgraph(feat.T,method='lnpbasic',k=10)
    for i in range(iteration):
        f = alpha*np.dot(W,f) + (1-alpha)*f0
    #f = f/np.tile(f.sum(axis=1),(2,1)).T

    f = torch.from_numpy(f).float()
    # f.requires_grad = True
    return f#F.sigmoid(f)

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
        x = self.gc1(g,x)
        x = self.gc2(g,x)
        return x

# 我们实现的基于RBF计算边权重的模型
class RBFNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RBFNN, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.beta = nn.Parameter(torch.Tensor(1).uniform_(
                0, 1), requires_grad=True)
        self.reset_params()
    
    def reset_params(self):
        nn.init.xavier_uniform_(self.fc.weight)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        e = torch.cosine_similarity(edges.src['z'], edges.dst['z'])
        e = -self.beta*(1-e)
        return {'e':e.unsqueeze(-1)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        # equation (1)
        g = g.local_var()
        h = F.dropout(h,p=0.6)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')

class VGAE(nn.Module):
    def __init__(self,g,nfeat,nhid,nclass):
        super(VGAE,self).__init__()
        self.mu = SAGEConv(nfeat, nhid, aggregator_type='gcn', feat_drop=0.5, bias=False, activation=F.relu)
        self.logvar = SAGEConv(nfeat, nhid, aggregator_type='gcn', feat_drop=0.5, bias=False, activation=F.relu)
        self.classifer =SAGEConv(nhid, nclass, aggregator_type='gcn', feat_drop=0.5, bias=False, activation=F.relu)
        self.reconstruct = SAGEConv(nhid, nfeat, aggregator_type='gcn', feat_drop=0.5, bias=False, activation=F.relu)
        # self.mu = GraphConv(nfeat, nhid, bias=False, activation=F.relu)
        # self.logvar = GraphConv(nfeat, nhid, bias=False, activation=F.relu)
        # self.classifer = GraphConv(nhid, nclass, bias=False, activation=F.relu)
        # self.reconstruct = GraphConv(nhid, nfeat, bias=False, activation=F.relu)
        self.g = g
        self.rbfnn = RBFNN(nhid, nclass)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self,x):
        x = F.dropout(x)
        g = self.g.local_var()
        mu = self.mu(g,x)
        logvar = self.logvar(g,x)
        rst = self.classifer(g,mu)
        z = self.reparameterize(mu,logvar)
        recons = torch.sigmoid(z.mm(z.t()))
        if model == 'RBF':
            rbfout = self.rbfnn(g,mu)
            return rst,rbfout,recons,mu,logvar
        else:
            return rst,recons,mu,logvar

g, features, labels, mask = load_cora_data()
adj = g.adjacency_matrix().to_dense()
idx_train = torch.LongTensor(range(140))
idx_val = torch.LongTensor(range(2208,2708))
idx_test = torch.LongTensor(range(140,1140))
if model == 'GCN':
    net = GCN(g,in_dim=features.size()[1],hidden_dim=8,out_dim=labels.max()+1)
else:
    net = VGAE(g,nfeat=features.size()[1],nhid=16,nclass=labels.max()+1)

optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=5e-4)

def KL(mu,logvar):
    return -0.5 / features.size(0) * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

def evaluate(feat,idx):
    with torch.no_grad():
        net.eval()
        if model == 'GCN':
            logits = net(feat)
            loss = F.cross_entropy(logits[idx],labels[idx])
        elif model == 'VGAE':
            logits, recons, mu, logvar = net(feat)
            loss = F.binary_cross_entropy(recons,adj) + F.cross_entropy(logits[idx],labels[idx]) + KL(mu,logvar)
        else:
            logits, assist, recons, mu, logvar = net(feat)
            loss = F.binary_cross_entropy(recons,adj) + F.cross_entropy(logits[idx],labels[idx]) + F.cross_entropy(assist[idx],labels[idx]) + KL(mu,logvar)
        
        acc = accuracy(logits[idx],labels[idx])
    
    return acc,loss.detach().numpy()

# main loop
dur = []
losses = []
for epoch in range(100):
    if epoch >= 5:
        t0 = time.time()
        # if losses[-1]>np.mean(losses[-6:-1]):
        #     print('early stop at epoch {}'.format(epoch))
        #     break
    
    if model == 'GCN':
        logits = net(features)
        loss = F.cross_entropy(logits[idx_train],labels[idx_train])
    elif model == 'VGAE':
        logits, recons, mu, logvar = net(features)
        loss = F.binary_cross_entropy(recons,adj) + F.cross_entropy(logits[idx_train],labels[idx_train]) + KL(mu,logvar)
    else:
        logits, assist, recons, mu, logvar = net(features)
        loss = F.binary_cross_entropy(recons,adj) + F.cross_entropy(logits[idx_train],labels[idx_train]) + F.cross_entropy(assist[idx_train],labels[idx_train]) + KL(mu,logvar)
    
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

def prob_lp(feat,label,nclass):
    y = torch.zeros(len(label),nclass)
    for i in range(len(label)):
        y[i,label[i]] = 1.0
    
    f0 = y.detach().numpy()
    feat = feat.detach().numpy()
    W = buildgraph(feat.T,method='lnpbasic',k=5)
    Q = np.eye(W.shape[0]) - W
    Qul = Q[140:1140,0:140]
    Quu = Q[140:1140,140:1140]
    f= np.linalg.solve(Quu,-Qul.dot(f0))
    # torch也有solve函数
    # X,LU = torch.solve(B,A)是方程AX=B的解
    # LU是A的LU分解
    return torch.from_numpy(f).float()#F.sigmoid(f)

# f = prob_lp(mu,labels[idx_train],labels.max()+1)
# print('prob label prop acc {:.4f}'.format(
#     accuracy(f,labels[idx_test]))) # Acc:0.6120
f = label_prop(mu,labels[idx_train],labels.max()+1) #Acc:0.7490
#f = label_prop(logits,labels[idx_train],labels.max()+1)
print('label prop acc {:.4f}'.format(
    accuracy(f[idx_test],labels[idx_test])))

'''
直接用二次规划求解W，或者用RBF核定义W，再放入IndGCN，
准确率低，loss为NaN，这一方法不可取。
IndGCN的实现本身没有问题，将既有的AugNormAdj放入IndGCN，
准确率0.8140

feat = mu.detach().numpy().T
feat = features.detach().numpy().T
#W = buildgraph(feat,method='lnpbasic',sigma=0.15,k=5)
feat = minmax_scale(feat,axis=1)
featprod = np.dot(feat.T,feat)
dmat,C = neighborhood(feat,k=10)
sigma = 1
wmat = np.exp(-dmat/(2*sigma**2))
C = C.T*C + np.eye(wmat.shape[0])
W = normalized(wmat) * C
#adj = adj.detach().numpy()
#W = normalized(adj)
W = sp.coo_matrix(W)
gh = DGLGraph()
gh.add_nodes(features.size(0))
gh.add_edges(W.row,W.col)
gh.edata['e'] = W.data

class IndLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=False,
                 activation=F.relu,
                 dropout=0.5):
        super(IndLayer, self).__init__()
        self.w = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()
        self._activation = activation
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, g, feat):
        graph = g.local_var()
        feat = self.dropout(feat)
        feat = self.w(feat)
        graph.ndata['h'] = feat
        graph.update_all(fn.src_mul_edge('h', 'e', 'm'),
                fn.sum('m', 'h'))
        rst = graph.ndata.pop('h')
        # 类似g.ndata.pop('h'),g.edata.pop('e')这样的写法，并不会减少内存的消耗
        rst = self._activation(rst)

        return rst

class IndGCN(nn.Module):
    def __init__(self,g,in_dim,hidden_dim,out_dim):
        super(IndGCN, self).__init__()
        self.gc1 = IndLayer(in_dim,hidden_dim)
        self.gc2 = IndLayer(hidden_dim,out_dim)
        self.g = g
    
    def forward(self,x):
        g = self.g.local_var()
        x = self.gc1(g,x)
        x = self.gc2(g,x)
        return x

def indevaluate(feat,idx):
    with torch.no_grad():
        net.eval()
        logits = net(feat)
        loss = F.cross_entropy(logits[idx],labels[idx])
        
        acc = accuracy(logits[idx],labels[idx])
    
    return acc,loss.detach().numpy()

net = IndGCN(gh,in_dim=features.size()[1],hidden_dim=8,out_dim=labels.max()+1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=5e-4)
# main loop
dur = []
losses = []
for epoch in range(100):
    if epoch >= 5:
        t0 = time.time()
        if losses[-1]>np.mean(losses[-6:-1]):
            print('early stop at epoch {}'.format(epoch))
            break
    
    logits = net(features)
    loss = F.cross_entropy(logits[idx_train],labels[idx_train])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 5:
        dur.append(time.time() - t0)
    
    acc_val,val_loss = indevaluate(features,idx_val)
    losses.append(val_loss)
    print("Epoch {} | Loss {:.4f} | Time(s) {:.4f} | Acc_Val {:.4f} | Loss_Val {:.4f}".format(
        epoch, loss.item(), np.mean(dur),acc_val,val_loss))

acc_test,_ = indevaluate(features,idx_test)
print('Acc_Test {:.4f}'.format(acc_test))
'''