"""
.. currentmodule:: dgl

Graph Classification Tutorial
=============================

**Author**: `Mufei Li <https://github.com/mufeili>`_,
`Minjie Wang <https://jermainewang.github.io/>`_,
`Zheng Zhang <https://shanghai.nyu.edu/academics/faculty/directory/zheng-zhang>`_.

In this tutorial, you learn how to use DGL to batch multiple graphs of variable size and shape. The 
tutorial also demonstrates training a graph neural network for a simple graph classification task.

Graph classification is an important problem
with applications across many fields, such as bioinformatics, chemoinformatics, social
network analysis, urban computing, and cybersecurity. Applying graph neural
networks to this problem has been a popular approach recently. This can be seen in the following reserach references: 
`Ying et al., 2018 <https://arxiv.org/abs/1806.08804>`_,
`Cangea et al., 2018 <https://arxiv.org/abs/1811.01287>`_,
`Knyazev et al., 2018 <https://arxiv.org/abs/1811.09595>`_,
`Bianchi et al., 2019 <https://arxiv.org/abs/1901.01343>`_,
`Liao et al., 2019 <https://arxiv.org/abs/1901.01484>`_,
`Gao et al., 2019 <https://openreview.net/forum?id=HJePRoAct7>`_).

"""

###############################################################################
# Simple graph classification task
# --------------------------------
# In this tutorial, you learn how to perform batched graph classification
# with DGL. The example task objective is to classify eight types of topologies shown here.
#
# .. image:: https://data.dgl.ai/tutorial/batch/dataset_overview.png
#     :align: center
#
# Implement a synthetic dataset :class:`data.MiniGCDataset` in DGL. The dataset has eight 
# different types of graphs and each class has the same number of graph samples.

from dgl.data import MiniGCDataset, TUDataset
# import matplotlib.pyplot as plt
import networkx as nx
# A dataset with 80 samples, each graph is
# of size [10, 20]
# dataset = MiniGCDataset(80, 10, 20)
# graph, label = dataset[0]
# fig, ax = plt.subplots()
# nx.draw(graph.to_networkx(), ax=ax)
# ax.set_title('Class: {:d}'.format(label))
# plt.show()

###############################################################################
# Form a graph mini-batch
# -----------------------
# To train neural networks efficiently, a common practice is to batch
# multiple samples together to form a mini-batch. Batching fixed-shaped tensor
# inputs is common. For example, batching two images of size 28 x 28
# gives a tensor of shape 2 x 28 x 28. By contrast, batching graph inputs
# has two challenges:
#
# * Graphs are sparse.
# * Graphs can have various length. For example, number of nodes and edges.
#
# To address this, DGL provides a :func:`dgl.batch` API. It leverages the idea that
# a batch of graphs can be viewed as a large graph that has many disjointed 
# connected components. Below is a visualization that gives the general idea.
#
# .. image:: https://data.dgl.ai/tutorial/batch/batch.png
#     :width: 400pt
#     :align: center
#
# Define the following ``collate`` function to form a mini-batch from a given
# list of graph and label pairs.

import dgl
import torch

cuda = True

torch.manual_seed(0)
if cuda:
    torch.cuda.manual_seed(0)

def compute_y(g):
    wmat = g.adjacency_matrix().to_dense()
    deg = torch.diag(torch.sum(wmat,dim=0))
    degpow = torch.pow(deg,-0.5)
    degpow[torch.isinf(degpow)] = 0
    lap = torch.mm(torch.mm(degpow,deg-wmat),degpow)
    e,v = torch.eig(lap,eigenvectors=True)
    g.ndata['y'] = F.relu(v[torch.argsort(e[:,0])[1]])

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_graph.to(torch.device('cuda:0'))
    return batched_graph, torch.tensor(labels).cuda()

###############################################################################
# The return type of :func:`dgl.batch` is still a graph. In the same way, 
# a batch of tensors is still a tensor. This means that any code that works
# for one graph immediately works for a batch of graphs. More importantly,
# because DGL processes messages on all nodes and edges in parallel, this greatly
# improves efficiency.
#
# Graph classifier
# ----------------
# Graph classification proceeds as follows.
#
# .. image:: https://data.dgl.ai/tutorial/batch/graph_classifier.png
#
# From a batch of graphs, perform message passing and graph convolution
# for nodes to communicate with others. After message passing, compute a
# tensor for graph representation from node (and edge) attributes. This step might 
# be called readout or aggregation. Finally, the graph 
# representations are fed into a classifier :math:`g` to predict the graph labels.
#
# Graph convolution layer can be found in the ``dgl.nn.<backend>`` submodule.

from dgl.nn.pytorch import GraphConv,GATConv,GINConv

###############################################################################
# Readout and classification
# --------------------------
# For this demonstration, consider initial node features to be their degrees.
# After two rounds of graph convolution, perform a graph readout by averaging
# over all node features for each graph in the batch.
#
# .. math::
#
#    h_g=\frac{1}{|\mathcal{V}|}\sum_{v\in\mathcal{V}}h_{v}
#
# In DGL, :func:`dgl.mean_nodes` handles this task for a batch of
# graphs with variable size. You then feed the graph representations into a
# classifier with one linear layer to obtain pre-softmax logits.

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import Set2Set

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, bias=True):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, bias=bias)
        # self.conv1 = GATConv(in_dim,hidden_dim,num_heads=4,feat_drop=0.3,attn_drop=0.2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=bias)
        # self.conv2 = GATConv(hidden_dim,hidden_dim,num_heads=4,feat_drop=0.3,attn_drop=0.2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gdata = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.ss = Set2Set(hidden_dim,10,2)
        self.classify = nn.Linear(hidden_dim, n_classes, bias=bias)
        self.hid = hidden_dim
        self.expand = False
        self.vec = GraphConv(hidden_dim, 1, bias=bias)
        self.dec = GraphConv(1, hidden_dim, bias=bias)
        # self.beta = nn.Parameter(torch.Tensor(1).uniform_(0, 1), requires_grad=True)

    def forward(self, g):
        # h = torch.eye(100)[g.ndata['node_labels'].squeeze()].float().cuda() # worse than using degree as node feat
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        # h = g.in_degrees().view(-1, 1).float().cuda()
        g.ndata['d'] = g.in_degrees().float().cuda()
        # g.ndata['d'] = self.beta*g.in_degrees().float().cuda() # worse, about 66%
        # g.ndata['d'] /= g.ndata['d'].norm() # worse, about 61%
        h = torch.eye(20)[g.in_degrees()].float().cuda() # better
        # h = g.ndata['node_labels'].float().cuda() # worse than using degree as node feat
        # Perform graph convolution and activation function.
        h = F.elu(self.bn1(self.conv1(g, h)),alpha=1.0)
        h = F.elu(self.bn2(self.conv2(g, h)),alpha=1.0)
        # h = F.relu(self.bn1(self.conv1(g, h).mean(dim=1))) # GATConv
        # h = F.relu(self.bn2(self.conv2(g, h).mean(dim=1))) # GATConv
        # before inputing h into a GraphConv layer
        g.ndata['h'] = h
        v = self.vec(g,h)
        g.ndata['d'] = v.squeeze()
        z = F.relu(self.dec(g,v))
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        ubg = dgl.unbatch(g)
        b = g.batch_size
        bnn = g.batch_num_nodes
        hid = self.hid
        if self.expand:
            bnnl = []
            bnnl.append(0)
            h_expand = torch.zeros(sum(bnn),b*hid)
            for i in range(b):
                bnnl.append(sum(bnn[0:i+1]))
            
            for i in range(b):
                h_expand[bnnl[i]:bnnl[i+1],i*hid:(i+1)*hid] = ubg[i].ndata['h']
            
            vec = torch.matmul(g.ndata['y'],h_expand)
            # vec = torch.matmul(g.in_degrees().float(),h_expand)
        else:
            hgl = []
            for i in range(b):
                vec = torch.matmul(ubg[i].ndata['d'],ubg[i].ndata['h'])
                hgl.append(vec)
            
            vec = torch.cat(hgl)

        mean = vec.mean()
        vec = F.relu(vec-mean)-F.relu(-vec-mean)
        hg = F.tanh(torch.reshape(vec,(b,hid)))
        # equals to:
        # hg = vec.view(-1,hid)
        # hg = vec.view(b,-1)
        # hg = self.ss(g,h).view(b,hid,2).mean(dim=-1)
        hg = F.tanh(self.bn3(self.gdata(hg)))
        if self.training:
            return z,F.tanh(self.classify(hg))
        else:
            return F.tanh(self.classify(hg))

###############################################################################
# Setup and training
# ------------------
# Create a synthetic dataset of :math:`400` graphs with :math:`10` ~
# :math:`20` nodes. :math:`320` graphs constitute a training set and
# :math:`80` graphs constitute a test set.

import torch.optim as optim
from torch.utils.data import DataLoader

dd = TUDataset('DD')

# Create training and test sets.
from dgl.data.utils import save_graphs,load_graphs,split_dataset
dataset = [(dd.__getitem__(i)[0],int(dd.__getitem__(i)[1])) for i in range(len(dd))]
'''
data = MiniGCDataset(1000, 10, 20)
g_list = [data[i][0] for i in range(1000)]
labels = [data[i][1] for i in range(1000)]
graph_labels={'glabel':torch.tensor(labels)}
save_graphs("graphclassification.pt", g_list, graph_labels)
glist, label_dict = load_graphs("graphclassification.pt")
# The edata and ndata of a graph can also be saved and loaded in this way.
dataset = [(glist[i],int(label_dict['glabel'][i])) for i in range(1000)]
'''
trainset,valset,testset = split_dataset(
    dataset,
    #[0.1,0,0.9],
    [0.9,0,0.1],
    shuffle=True,random_state=0)

# for i in range(len(trainset)):
#     compute_y(trainset[i][0])

# for i in range(len(testset)):
#     compute_y(testset[i][0])

# Use PyTorch's DataLoader and the collate function
# defined before.
data_loader = DataLoader(trainset, batch_size=16, shuffle=True,
                         collate_fn=collate)

# Create model
model = Classifier(20, 64, 2).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
model.train()

epoch_losses = []
for epoch in range(1000):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(data_loader):
        z,prediction = model(bg)
        loss = F.cross_entropy(
            prediction, label) + F.mse_loss(
                z, bg.ndata['h'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    # epoch_loss /= (iter + 1)
    # print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    # epoch_losses.append(epoch_loss)
    if epoch % 10 == 0:
        with torch.no_grad():
            test_X, test_Y = map(list, zip(*testset))
            test_bg = dgl.batch(test_X)
            test_bg.to(torch.device('cuda:0'))
            _,test_feat = model(test_bg)
            probs_Y = torch.softmax(test_feat, 1).cpu()
            test_Y = torch.tensor(test_Y).float().view(-1, 1)
            sampled_Y = torch.multinomial(probs_Y, 1)
            argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
            print('Epoch {}, Accuracy of argmax on test: {:4f}%'.format(
                epoch,(test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

###############################################################################
# The learning curve of a run is presented below.

# plt.title('cross entropy averaged over minibatches')
# plt.plot(epoch_losses)
# plt.show()

###############################################################################
# The trained model is evaluated on the test set created. To deploy
# the tutorial, restrict the running time to get a higher
# accuracy (:math:`80` % ~ :math:`90` %) than the ones printed below.

model.eval()

import numpy as np
from sklearn.preprocessing import minmax_scale
from math import sqrt

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
    deg = np.diag(np.sum(wmat,axis=0))
    degpow = np.power(deg,-0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow,wmat),degpow)

    return W

def buildgraph(feat,method='lp',sigma=0.01,k=5):
    feat = minmax_scale(feat,axis=1)
    featprod = np.dot(feat.T,feat)
    dmat,C = neighborhood(feat,k)
    wmat = np.exp(-dmat/(2*sigma**2))
    W = normalized(wmat)
    
    return W

def svt(A,lam=2):
    U,S,V = np.linalg.svd(A,full_matrices=0)
    sigmam = S[0]
    S = np.maximum(0,S-lam)
    S = np.diag(S)
    
    return np.dot(np.dot(U,S),V),sigmam

def lasso_prox(A,lam):
    return np.maximum(0,A-lam) + np.minimum(0,A+lam)

def admm(feat):
    feat = minmax_scale(feat,axis=1)
    tau = svt(feat)[1]**2
    e = np.zeros(feat.shape)
    z = np.zeros((feat.shape[1],feat.shape[1]))
    t = 1
    s = tau
    sigmam0 = np.inf
    sigmaml = []
    for i in range(100):
        e0 = e
        z0 = z
        t0 = t
        bare = e + (t0-1)*(e-e0)/t
        barz = z + (t0-1)*(z-z0)/t
        ge = bare + (feat - np.dot(feat,barz) - bare)
        gz = barz + np.dot(feat.T,(feat - np.dot(feat,barz) - bare))/tau
        z,sigmam = svt(gz,s/tau)
        e = lasso_prox(ge,s/tau)
        sigmaml.append(sigmam)
        if i > 5 and sigmam > sum(sigmaml[-1:-6])/5:
            print('stop:',i)
            break # early stopping

        if sigmam > sigmam0:
            s = s*0.5
            #z,_ = svt(gz,s)
        
        t = 0.5*(1+sqrt(4*t0**2+1))
    
    return normalized(z)

test_X, test_Y = map(list, zip(*testset))
test_bg = dgl.batch(test_X)
test_bg.to(torch.device('cuda:0'))
test_feat = model(test_bg)
probs_Y = torch.softmax(test_feat, 1).cpu()
test_Y = torch.tensor(test_Y).float().view(-1, 1)
"""
train_X, train_Y = map(list, zip(*trainset))
train_bg = dgl.batch(train_X)
test_X, test_Y = map(list, zip(*testset))
test_bg = dgl.batch(test_X)
test_Y = torch.tensor(test_Y).float().view(-1, 1)
train_feat = model(train_bg)
test_feat = model(test_bg)
feat = torch.cat((train_feat,test_feat),dim=0).detach().numpy()
W = buildgraph(feat.T)
# W = admm(feat.T)
f0 = np.zeros((len(trainset)+len(testset),8))
for i in range(len(trainset)):
    f0[i,trainset[i][1]] = 1

f = f0
for e in range(50):
    f = 0.99*np.dot(W,f) + 0.01*f0

lp_Y = torch.Tensor(f[-len(testset):])
probs_Y = torch.softmax(test_feat, 1)
print('==labelprop==')
# lsampled_Y = torch.multinomial(lp_Y, 1)
largmax_Y = torch.max(lp_Y, 1)[1].view(-1, 1)
# print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
#     (test_Y == lsampled_Y.float()).sum().item() / len(test_Y) * 100))
print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
    (test_Y == largmax_Y.float()).sum().item() / len(test_Y) * 100))
print('==basic==') """
sampled_Y = torch.multinomial(probs_Y, 1)
argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
    (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
    (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

###############################################################################
# The animation here plots the probability that a trained model predicts the correct graph type.
#
# .. image:: https://data.dgl.ai/tutorial/batch/test_eval4.gif
#
# To understand the node and graph representations that a trained model learned,
# we use `t-SNE, <https://lvdmaaten.github.io/tsne/>`_ for dimensionality reduction
# and visualization.
#
# .. image:: https://data.dgl.ai/tutorial/batch/tsne_node2.png
#     :align: center
#
# .. image:: https://data.dgl.ai/tutorial/batch/tsne_graph2.png
#     :align: center
#
# The two small figures on the top separately visualize node representations after one and two
# layers of graph convolution. The figure on the bottom visualizes
# the pre-softmax logits for graphs as graph representations.
#
# While the visualization does suggest some clustering effects of the node features,
# you would not expect a perfect result. Node degrees are deterministic for
# these node features. The graph features are improved when separated.
#
# What's next?
# ------------
# Graph classification with graph neural networks is still a new field.
# It's waiting for people to bring more exciting discoveries. The work requires 
# mapping different graphs to different embeddings, while preserving
# their structural similarity in the embedding space. To learn more about it, see 
# `How Powerful Are Graph Neural Networks? <https://arxiv.org/abs/1810.00826>`_ a research paper  
# published for the International Conference on Learning Representations 2019.
#
# For more examples about batched graph processing, see the following:
#
# * Tutorials for `Tree LSTM <https://docs.dgl.ai/tutorials/models/2_small_graph/3_tree-lstm.html>`_ and `Deep Generative Models of Graphs <https://docs.dgl.ai/tutorials/models/3_generative_model/5_dgmg.html>`_
# * An example implementation of `Junction Tree VAE <https://github.com/dmlc/dgl/tree/master/examples/pytorch/jtnn>`_
