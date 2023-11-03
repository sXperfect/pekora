import torch

def compute_gram(
    P_rows:torch.Tensor,
    P_cols:torch.Tensor
):
    G = (P_rows*P_cols).sum(axis=1)
    return G

def compute_euc_dist_square(
    P_rows:torch.Tensor,
    P_cols:torch.Tensor,
    G=None
):

    D = torch.square(P_rows).sum(axis=1)
    D += torch.square(P_cols).sum(axis=1)
    if G is None:
        D -= 2*(P_rows*P_cols).sum(axis=1)
    else:
        D -= 2*G

    return D

def compute_trace_P(
    P:torch.tensor,
    reduction='mean',
):
    T = P.square()
    if reduction == 'mean':
        T = T.mean()
    elif reduction == 'sum':
        T = T.sum()
    else:
        raise NotImplementedError('Not yet implemented')

    return T

def compute_contacts(
    D,
    alpha
):
    C = D**(1/alpha)
    
    return C