import numpy as np
import pandas as pd
from scipy import stats
from .. import const
from .. import loaders
from .. import utils

DELIMITER = "\t"

def comp_superrec_spearmanr(
    chr1_region:str,
    resolution:int,
    balancing:str,
    input:str,
    points:str,
    chr2_region:str=None,
):

    count_df = loaders.load_3c_data(
        input,
        chr1_region,
        resolution,
        balancing=balancing,
        chr2_region=chr2_region,
        ret_df=True
    )

    row_ids = count_df[const.ROW_IDS_COLNAME].to_numpy()
    col_ids = count_df[const.COL_IDS_COLNAME].to_numpy()
    counts = count_df[const.COUNTS_COLNAME].to_numpy()

    #? Create mapping from row/col ids to points
    #? Reason: not all loci are valid, yet the points contains only valid points
    unique_ids = np.unique([row_ids, col_ids])
    mapping = np.searchsorted(unique_ids, np.arange(unique_ids.max()+1))

    new_row_ids = mapping[row_ids]
    new_col_ids = mapping[col_ids]

    points = pd.read_csv(
        points,
        # delimiter=DELIMITER,
        delim_whitespace=True, #? Delimiter is all possible whitespace
        header=None,
        names=["X", "Y", "Z"],
        index_col=None,
    ).to_numpy()

    D = utils.edm_numpy.compute_D_from_Ps(
        points,
        new_row_ids,
        new_col_ids
    )

    res = stats.spearmanr(counts, D)
    corr = res.correlation
    
    #? Data_ratio is one because it involves mapping from original point indices to
    #? the new point indices
    data_ratio = 1.0

    print(f"Region (Chromosome):{chr1_region}; Spearman R:{corr:.03f}; Data Ratio:{data_ratio:.03f}")