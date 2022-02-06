import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cvxopt import matrix,solvers
solvers.options['show_progress'] = False

model = 'RBF'
torch.manual_seed(72)
'''
当使用二次规划的迭代方法，
也就是model='QUAR'时，
准确率只有50%左右，一般不使用。
model可以为
'GAT','AGNN','MPNN','RBF'
'''

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.beta = nn.Parameter(torch.Tensor(1).uniform_(
                0, 1), requires_grad=True)
        self.mpnn = nn.Linear(out_dim, 1)
        self.reset_params()
    
    def reset_params(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_fc.weight)
        nn.init.xavier_uniform_(self.mpnn.weight)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        if model == 'GAT':
            # GAT
            z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
            a = self.attn_fc(z2)
            return {'e': F.leaky_relu(a)}
        elif model == 'AGNN':
            # AGNN: cosine sim
            e = torch.cosine_similarity(edges.src['z'], edges.dst['z'])
            e = self.beta * e
            return {'e':e.unsqueeze(-1)}
        elif model == 'MPNN':
            e = self.mpnn(edges.src['z'])
            return {'e':e}
        else:
            # e = torch.cosine_similarity(edges.src['z'], edges.dst['z'])
            # e = -self.beta*(1-e)
            e = torch.norm(edges.src['z']-edges.dst['z'], dim=1)
            e = -self.beta*e
            if model == 'QUAR':
                d = torch.norm(edges.src['z']-edges.dst['z'], p=1, dim=1)
                return {'e':e.unsqueeze(-1),'d':d}
            else:
                return {'e':e.unsqueeze(-1)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        if model == 'QUAR':
            return {'z': edges.src['z'], 'e': edges.data['e'], 'd': edges.data['d']}
        else:
            return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        if model == 'QUAR':
            d = nodes.mailbox['d']
            k = nodes.mailbox['d'].size(1)
            gram = torch.mm(d.t(),d)
            #gram = torch.pow(torch.mm(d.t(),d),0.5)
            #gram[torch.isnan(gram)] = 0.0
            mu = 2
            '''
            z = nodes.mailbox['z']
            gram = torch.zeros(k,k)
            for i in range(k):
                for j in range(k):
                    gram[i,j] = torch.norm(z[:,i,:]-z[:,j,:])
            
            gram = gram.detach().numpy()
            gram = np.power(gram,2) + mu*np.eye(gram.shape[0])
            #print(gram.shape)
            Q = 2 * matrix(gram)
            p = matrix(np.zeros(k))  # 代表一次项的系数
            G = -1 * matrix(np.eye(k))  # G和h代表GX+s = h，s>=0,表示每一个变量x均大于零
            h = p
            A = matrix(np.ones(k),(1,k))
            b = matrix(1.0)                                                    # AX = b
            sol = solvers.qp(Q, p, G, h, A, b)
            alpha = torch.from_numpy(
                np.array(sol['x'])).float()
            '''
            e = torch.ones(k,1)
            alpha = mu*alpha/(gram.matmul(alpha)+mu*e.mm(e.t()).matmul(alpha))

        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        self.g = self.g.local_var()
        h = F.dropout(h,p=0.6)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
        # 类似g.ndata.pop('h'),g.edata.pop('e')这样的写法，并不会减少内存的消耗

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        self.n_heads = num_heads
        for i in range(self.n_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            # return torch.mean(torch.stack(head_outs))
            return sum(head_outs)/self.n_heads

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, merge='mean'):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h

from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx

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

import time
from dgl.nn.pytorch.conv import GraphConv
from sklearn.metrics import f1_score

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

g, features, labels, mask = load_cora_data()
idx_train = torch.LongTensor(range(140))
idx_val = torch.LongTensor(range(140,640))
idx_test = torch.LongTensor(range(1708,2708))
# create the model, 2 heads, each head has hidden size 8
net = GAT(g,
          in_dim=features.size()[1],
          hidden_dim=8,
          out_dim=labels.max()+1,
          num_heads=8)

# net = GCN(g,in_dim=features.size()[1],hidden_dim=8,out_dim=labels.max()+1)
# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

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
for epoch in range(100):
    if epoch >= 5:
        t0 = time.time()
        if losses[-1]>np.mean(losses[-6:-1]):
            print('early stop at epoch {}'.format(epoch))
            break

    logits = net(features)
    # logp = F.log_softmax(logits, 1)
    # loss = F.nll_loss(logp[mask], labels[mask])
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