import numpy as np
import pandas as pd
from scipy import stats
from .. import const
from .. import loaders
from .. import utils

def comp_pekora_spearmanr(
    chr1_region:str,
    resolution:int,
    balancing:str,
    input_fpath:str,
    points_fpath:str,
    chr2_region:str=None,
):

    count_df = loaders.load_3c_data(
        input_fpath,
        chr1_region,
        resolution,
        balancing=balancing,
        chr2_region=chr2_region,
        ret_df=True
    )

    row_ids = count_df[const.ROW_IDS_COLNAME].to_numpy()
    col_ids = count_df[const.COL_IDS_COLNAME].to_numpy()
    counts = count_df[const.COUNTS_COLNAME].to_numpy()
    
    points = np.load(
        points_fpath
    )
    
    D = utils.edm_numpy.compute_D_from_Ps(
        points,
        row_ids,
        col_ids
    )
    
    res = stats.spearmanr(counts, D)
    corr = -res.statistic
    
    #? data_ratio is one because we do not remove explicitly any region
    #? There is no additional mapping/filtering
    data_ratio = 1.0

    #? data_ratio is one because we do not remove explicitly any region
    print(f"Region (Chromosome):{chr1_region}; Spearman R:{corr:.03f}; Data Ratio:{data_ratio:.03f}")