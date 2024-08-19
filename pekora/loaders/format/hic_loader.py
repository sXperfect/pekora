import numpy as np
import pandas as pd
from .. import const

def load_hic(
    path,
    chr1_region:str,
    resolution:int,
    balancing:str=None,
    chr2_region:str=None,
    ret_df=False,
):
    
    if balancing is None:
        balancing = "NONE"
        
    import hicstraw
    recs = hicstraw.straw(
        'observed',
        balancing,
        path,
        chr1_region, 
        chr1_region if chr2_region is None else chr2_region,
        'BP', resolution
    )

    row_ids = np.empty(len(recs), dtype=np.uint32)
    col_ids = np.empty(len(recs), dtype=np.uint32)
    counts = np.empty(len(recs), dtype=np.float64)

    #TODO: Do we need offset of 1 so that row_ids += -1
    for i, rec in enumerate(recs):
        row_ids[i] = int(rec.binX/resolution)
        col_ids[i] = int(rec.binY/resolution)
        counts[i] = rec.counts

    if ret_df:
        df = pd.DataFrame()
        df.insert(0, const.ROW_IDS_COLNAME, row_ids)
        df.insert(1, const.COL_IDS_COLNAME, col_ids)
        df.insert(2, const.COUNTS_COLNAME, counts)
        return df
    else:
        return row_ids, col_ids, counts