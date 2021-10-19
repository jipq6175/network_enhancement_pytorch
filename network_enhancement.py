

# Network Enhancement implemeentation in pytorch
# Author =  Yen @ ReviveMed 


import torch
import numpy as np

# for stability
EPS = 2e-16

# DN Normallization Function: Correct
def DN(w, tp='ave'): 

    assert tp in ['ave', 'gph']
    assert w.shape[0] == w.shape[1]
    n = w.shape[0]

    W = w.detach().clone() # not changing the original tensor
    W *= n
    D = W.abs().sum(axis=1) + EPS

    if tp == 'ave':
        D = 1 / D
        D = D.diag_embed().to_sparse()
        wn = D.matmul(W)
    elif tp == 'gph':
        D = 1 / D.sqrt()
        D = D.diag_embed().to_sparse()
        wn = D.matmul(W.matmul(D))
    else: 
        NotImplementedError()

    return wn

    



# Transition fields: Correct
# The output is doubly stochastic
def TransitionFields(w): 

    assert w.shape[0] == w.shape[1]
    n = w.shape[0]

    W = w.detach().clone() # not changing the original tensor
    zero_index = torch.where(W.sum(axis=1) == 0.0)[0]
    W *= n
    W = DN(W, tp='ave')
    
    d = (W.abs().sum(axis=0) + EPS).sqrt()

    W /= d.repeat((n, 1))
    W = W.matmul(W.t())
    W[zero_index, :] = 0.0
    W[:, zero_index] = 0.0
    return W





# Dominate Set
# This is the most error-prone function due to inconsistent index handling between matlab and python
def DominateSet(maff, nr_knn):

    assert maff.shape[0] == maff.shape[1]
    n = maff.shape[0]
    assert  nr_knn <= n

    sorted_maff, indices = torch.sort(maff, dim=1, descending=True)
    res = sorted_maff[:, 0:nr_knn].t().contiguous().view(1, -1)
    
    #  get the indeces for assignments
    inds = torch.tensor(range(n)).repeat((1, nr_knn))
    loc = indices[:, 0:nr_knn].t().contiguous().view(1, -1)
    assert res.shape[1] == inds.shape[1] == loc.shape[1] == n * nr_knn
    # they are of shape 1 x (n * nr_knn)

    ix = torch.cat((inds, loc), dim=0)
    pnn = torch.sparse_coo_tensor(ix, res.view(-1), size=[n, n])
    pnn = 0.5 * (pnn + pnn.t())

    return pnn.to_dense()





# Network Enhancement
def NetworkEnhancement(w, order=2, K=20, alpha=0.9): 
    
    assert w.shape[0] == w.shape[1]
    assert torch.all(w.t() == w)
    n = w.shape[0]
    
    # update k based on dim and inputs
    k = int(min(K, np.ceil(n/10)))

    W = w.detach()
    W *= (1 - torch.eye(n))
    zero_index = torch.where(W.abs().sum(axis=0) > 0.0)[0]
    W0 = W[np.ix_(zero_index, zero_index)]
    W = DN(W0, tp='ave')
    W = 0.5 * (W + W.t())

    DD = W0.abs().sum(axis=0)


    if W.unique().shape[0] == 2: 
        P = W
    else: 
        P = DominateSet(W.abs(), min(k, n-1)) * W.sign()
    
    P += torch.eye(n) + P.abs().sum(axis=0).diag_embed()
    P = TransitionFields(P)
    # check points succeed so far

    lambdas, evectors = torch.linalg.eigh(P)
    d = lambdas - EPS
    d = (1 - alpha) * d / (1 - alpha * d ** order)
    D = d.diag_embed()
    W = evectors.matmul(D).matmul(evectors.inverse()) 
    # W here is doubly stochastic

    W = (W * (1 - torch.eye(n))) / (1 - W.diag()).repeat(n, 1).t()


    D = torch.diag(DD).to_sparse()
    W = D.matmul(W)
    # print(W)

    W[W < 0] = 0
    W = 0.5 * (W + W.t())
    
    rlt = torch.zeros((n, n), dtype=W.dtype)
    rlt[np.ix_(zero_index, zero_index)] = W

    return rlt





if __name__ == '__main__':


    from scipy.io import loadmat
    
    # load data from .mat
    d = loadmat('./Butterfly_Network/Raw_butterfly_network.mat')
    W_butterfly0 = torch.tensor(d['W_butterfly0'], dtype=torch.double)
    labels = torch.tensor(d['labels'], dtype=torch.int)

    W_butterfly_NE = NetworkEnhancement(W_butterfly0)

    pass