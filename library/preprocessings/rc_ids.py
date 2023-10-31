import typing as t
import numpy as np
import pandas as pd
from .. import const

def project_row_col_ids(
    df
):
    
    row_ids = df.loc[:, const.ROW_IDS_COLNAME].to_numpy()
    col_ids = df.loc[:, const.COL_IDS_COLNAME].to_numpy()

    ids = np.concatenate([
        row_ids,
        col_ids
    ])
    
    unique_ids = np.unique(ids)
    N = unique_ids[-1] + 1
    
    mapping_ids = np.zeros(N, dtype=int)
    mapping_ids[unique_ids] = np.arange(len(unique_ids))
    
    df.loc[:, const.ROW_IDS_COLNAME] = mapping_ids[row_ids]
    df.loc[:, const.COL_IDS_COLNAME] = mapping_ids[col_ids]
    
    return df