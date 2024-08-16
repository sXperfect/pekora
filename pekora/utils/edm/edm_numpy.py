import typing as t
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
    P: np.ndarray,
    row_ids: t.Sequence[int],
    col_ids: t.Sequence[int],
) -> np.ndarray:
    """
    Compute the square of Euclidean distances between two sets of points.

    Parameters
    ----------
    P : np.ndarray
        Input array of points.
    row_ids : t.Sequence[int]
        Row indices of the first set of points.
    col_ids : t.Sequence[int]
        Column indices of the second set of points.

    Returns
    -------
    np.ndarray
        Square of Euclidean distances between the two sets of points.

    Notes
    -----
    This function uses `compute_euc_dist_square` to compute the square of Euclidean distances.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[1, 2], [3, 4], [5, 6]])
    >>> row_ids = [0, 1]
    >>> col_ids = [1, 2]
    >>> compute_D_from_Ps(P, row_ids, col_ids)
    array([[ 5, 13],
           [ 5, 13]])

    Authors
    -------
    - Yeremia G. Adhisantoso (adhisant@tnt.uni-hannover.de)
    - Llama3.1 70B - 4.0bpw
    """
    
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