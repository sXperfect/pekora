import numpy as np

def compute_gram(
    P_rows:np.ndarray,
    P_cols:np.ndarray,
):
    G = (P_rows*P_cols).sum(axis=1)
    return G

def compute_euc_dist_square(
    P_rows:np.ndarray,
    P_cols:np.ndarray,
    G=None
):

    D = np.square(P_rows).sum(axis=1)
    D += np.square(P_cols).sum(axis=1)
    if G is None:
        D -= 2*(P_rows*P_cols).sum(axis=1)
    else:
        D -= 2*G

    return D

def compute_trace_P(
    P:np.ndarray,
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

def compute_D_from_Ps(
    P,
    row_ids,
    col_ids,
):
    P_rows = P[row_ids, :]
    P_cols = P[col_ids, :]

    D = compute_euc_dist_square(P_rows, P_cols)

    return D

def compute_contacts(
    D,
    alpha
):
    C = D**(1/alpha)
    
    return C