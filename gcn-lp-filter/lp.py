import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import classification_report,roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc,confusion_matrix
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

# from sklearn.semi_supervised import LabelPropagation # alpha=1
# from sklearn.semi_supervised.label_propagation import LabelSpreading # alpha 可设置
# sklearn中的LabelPropagation效果不好

np.random.seed(42)
N = 500 # 每一类的数量
stdev = 0.1 # 当标准差增大时，样本点的数量也要随着一并增大

def twomoon_data():
    x1 = np.linspace(0,1,N)#+stdev*np.random.randn(1,N)
    x2 = np.linspace(0.5,1.5,N)#+stdev*np.random.randn(1,N)
    y1 = np.sin(np.pi*x1)-0.2+stdev*np.random.randn(1,N)
    y2 = -np.sin(np.pi*(x2-0.5))+0.2+stdev*np.random.randn(1,N)
    # s1 = np.power(x1,2) + np.power(y1,2)
    # s2 = np.power(x2,2) + np.power(y2,2)
    l = np.hstack((np.zeros(N),np.ones(N)))
    l[[1,-1]] = l[[-1,1]]
    f0 = np.zeros((2,2*N))
    f0[0,0] = 1.0
    f0[1,1] = 1.0
    x = np.hstack((x1,x2))
    y = np.hstack((y1,y2))
    feat = np.vstack((x,y))
    feat[:,[1,-1]] = feat[:,[-1,1]]
    
    return feat,l,f0
    
def optimize_w(i,k,feat,dsort):
    dist = []
    for j in dsort[i]:
        dist.append(feat.T[i]-feat.T[j])
    
    dist = np.array(dist)
    gram = np.dot(dist,dist.T)
    Q = 2 * matrix(gram)
    p = matrix(np.zeros(k))  # 代表一次项的系数
    G = -1 * matrix(np.eye(k))  # G和h代表GX+s = h，s>=0,表示每一个变量x均大于零
    h = p
    A = matrix(np.ones(k),(1,k))
    b = matrix(1.0)                                                    # AX = b
    sol = solvers.qp(Q, p, G, h, A, b)
    w = np.zeros(2*N)
    ginv = np.linalg.inv(gram+0.01*np.eye(gram.shape[0]))
    ww = ginv.sum(axis=0)/ginv.sum()
    for j in range(k):
        w[dsort[i,j]] = sol['x'][j]
        #w[dsort[i,j]] = ww[j]
    
    return w

def buildgraph(feat,method='lp',sigma=0.05,k=10):
    feat = minmax_scale(feat,axis=1)
    featprod = np.dot(feat.T,feat)
    smat = np.tile(np.diag(featprod),(2*N,1))
    dmat = smat + smat.T - 2*featprod
    wmat = np.exp(-dmat/(2*sigma**2))
    deg = np.diag(np.sum(wmat,axis=0))
    degpow = np.power(deg,-0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow,wmat),degpow)
    '''
    if method == 'lnpbasic':
        w = []
        dsort = np.argsort(dmat)[:,1:k+1]
        for i in range(2*N):
            w.append(optimize_w(i,k,feat,dsort))
        
        W = np.array(w)
    else:
        dsort = np.argsort(dmat)[:,1:k+1]
        C = np.zeros((2*N,2*N))
        e = np.ones(2*N)
        mu = 2
        iteration = 100
        W = np.ones((2*N,2*N))/(2*N-1) - np.eye(2*N)
        for i in range(2*N):
            for j in dsort[i]:
                C[i,j] = 1.0
        
        for i in range(iteration):
            W = W * (featprod+mu*np.dot(e,e.T)) / (C*W*featprod+mu*C*W*np.dot(e,e.T))
        
        W[np.isnan(W)] = 0
        W = W/k
    '''
    return W

def label_prop(W,f0,alpha=0.99,iteration=100,method='basic'):
    f = f0.T
    if method == 'basic':
        for i in range(iteration):
            f = alpha*np.dot(W,f) + (1-alpha)*f0.T
    else:
        fl = f[0:2,:]
        Q = np.eye(2*N) - W
        # if use 2nd-order IGMRF then: 
        #Q = np.dot(Q,Q)
        Quu = Q[2:2*N,2:2*N]
        Qul = Q[2:2*N,0:2]
        fu = np.linalg.solve(Quu,-np.dot(Qul,fl))
        f = np.vstack((fl,fu))
    
    f = f/np.tile(f.sum(axis=1),(2,1)).T
    return f

def draw(feat,y):
    posx = []
    posy = []
    negx = []
    negy = []
    for i in range(len(y)):
        if y[i] == 1:
            posx.append(feat.T[i,0])
            posy.append(feat.T[i,1])
        else:
            negx.append(feat.T[i,0])
            negy.append(feat.T[i,1])
    
    plt.figure()
    plt.plot(posx,posy,'or')
    plt.plot(negx,negy,'ob')
    plt.plot(1.5,0.2,'or',label='Pos samples')
    plt.plot(0,-0.2,'ob',label='Neg samples')
    plt.legend()
    plt.show()

def binarymetrics(TP,TN,FP,FN):
    Sn = TP/(TP+FN) # Sp=Rec
    Sp = TN/(TN+FP)
    Acc = (TN+TP)/(TN+TP+FN+FP)
    Pre = TP/(TP+FP)
    F1 = 2*(Pre*Sn)/(Pre+Sn)
    Mcc = (TP*TN-FP*FN)/np.sqrt((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP))
    print('Sn={:.4f},Sp={:.4f},Acc={:.4f},Pre={:.4f},F1={:.4f},Mcc={:.4f}'.format(Sn,Sp,Acc,Pre,F1,Mcc))

def svm(feat):
    #clf = SVC(C=0.1,kernel='rbf',degree=2,gamma=0.1)
    clf = KNeighborsClassifier(n_neighbors=1)
    #clf = LabelSpreading(gamma=800,n_neighbors=4,alpha=0.99,max_iter=100)
    datamat = np.array([[0,-0.2],[1.5,0.2]])
    clf.fit(datamat,[0,1])
    return clf.predict(feat.T)

feat,y_true,f0 = twomoon_data()
W = buildgraph(feat,method='lnpbasic',sigma=0.05,k=5)
f = label_prop(W,f0,method='basic')
y_pred = np.argmax(f,axis=1)
#y_pred = svm(feat)
fpr,tpr,rocth = roc_curve(y_true,y_pred)
precision,recall,prth = precision_recall_curve(y_true,y_pred)
#y_pred = f[:,1]
cr = confusion_matrix(y_true,y_pred)
print(classification_report(y_true,y_pred,digits=4))
binarymetrics(cr[1,1],cr[0,0],cr[0,1],cr[1,0])
#print(roc_auc_score(y_true,y_pred))
print('AUROC {:.4f} AUPR {:.4f}'.format(auc(fpr,tpr),auc(recall,precision)))
draw(feat,y_pred)
'''
plt.figure()
plt.plot(fpr,tpr)
plt.show()
y_score = f[:,1]
fpr,tpr,thresh = roc_curve(y_true,y_score,pos_label=1.0)
plt.figure()
plt.plot(fpr,tpr)
plt.show()

x,y = twomoon_data()
plt.figure()
plt.plot(x,y,'or')
plt.show()
'''