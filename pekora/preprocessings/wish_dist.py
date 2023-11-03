import numpy as np
import pandas as pd
from .. import const

def comp_sparse_wish_dist_coo(
    row_ids:np.array,
    col_ids:np.array,
    C_vals:np.array,
    alpha=-0.25
):
    """Create sparse form of Contact and Euclidean Distance matrices.
    This function removes the diagonal content of the matrix.

    Parameters
    ----------
    row_ids : np.array
        _description_
    col_ids : np.array
        _description_
    C_vals : np.array
        _description_
    alpha : float, optional
        Conversion factor from contact matrix to Euclidean distance matrix, by default -0.25

    Returns
    -------
    _type_
        _description_
    """
       
    #? Remove entries in the main diagonal as euclidean distance would be zero
    nondiag_mask = row_ids != col_ids
    row_ids = row_ids[nondiag_mask]
    col_ids = col_ids[nondiag_mask]
    C_vals = C_vals[nondiag_mask]
    
    D_vals = np.power(C_vals, alpha)
    
    return row_ids, col_ids, C_vals, D_vals

def comp_sparse_wish_dist(
    df:pd.DataFrame,
    alpha=-0.25,
    na_inf_val:float=None,
)-> pd.DataFrame:
    
    """Create sparse form of Contact and Euclidean Distance matrices.
    This function removes the diagonal content of the matrix.

    Parameters
    ----------
    df: pd.DataFrame
        _description_
    alpha : float, optional
        Conversion factor from contact matrix to Euclidean distance matrix, by default -0.25

    Returns
    -------
    _type_
        _description_
    """
       
    #? Remove entries in the main diagonal as euclidean distance would be zero
    nondiag_mask = df[const.ROW_IDS_COLNAME] != df[const.COL_IDS_COLNAME]
    df = df.loc[nondiag_mask]
    
    #? Insert Euclidean distance to the dataframe as a new column
    dist_arr = np.power(df[const.COUNTS_COLNAME], alpha)
    
    na_inf_mask = np.logical_or(np.isinf(dist_arr), np.isnan(dist_arr))
    if na_inf_mask.any():
        if na_inf_val is not None:
            dist_arr[na_inf_mask] = na_inf_val
        else:
            dist_arr = dist_arr[~na_inf_mask]
            df = df.loc[~na_inf_mask, :]
        
    df.insert(
        df.shape[1], 
        const.DIST_COLNAME, 
        dist_arr
    )
    
    return df