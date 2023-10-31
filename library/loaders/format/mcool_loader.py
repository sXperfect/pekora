import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import cooler
from .. import const

def load_mcool(
    path,
    chr1_region:str,
    resolution:int,
    balancing:str=None,
    chr2_region:str=None,
    ret_df=False,
):
    if balancing is None:
        balancing = False

    #? Load data from cooler with normalization applied
    c = cooler.Cooler(path + f"::resolutions/{resolution}")
    C_coo:coo_matrix = (c
        .matrix(balance=balancing, sparse=True)
        .fetch(
            chr1_region,
            chr1_region if chr2_region is None else chr2_region,
        )
    )

    #? Get only the diagonal and upper triangle part
    row_ids = C_coo.row
    col_ids = C_coo.col
    counts = C_coo.data
    mask = row_ids <= col_ids
    if ~mask.all():
        row_ids = row_ids[mask]
        col_ids = col_ids[mask]
        counts = counts[mask]

    #? Remove all NaNs and zeros
    mask = np.logical_and(~np.isnan(counts), counts != 0)
    if ~mask.all():
        row_ids = row_ids[mask]
        col_ids = col_ids[mask]
        counts = counts[mask]
        
    if ret_df:
        df = pd.DataFrame()
        df.insert(0, const.ROW_IDS_COLNAME, row_ids)
        df.insert(1, const.COL_IDS_COLNAME, col_ids)
        df.insert(2, const.COUNTS_COLNAME, counts)
        return df
    else:
        return row_ids, col_ids, counts
