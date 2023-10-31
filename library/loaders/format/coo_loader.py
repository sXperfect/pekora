import numpy as np
from .csv_loader import load_csv
from .. import const

def load_coo(
    path,
    chr1_region:str,
    resolution:int,
    balancing:str="NONE",
    chr2_region:str=None,
    ret_df=False,
):
    #TODO: check the correctness of CHR
    assert np.issubdtype(type(resolution), np.integer), \
        "Resolution must be an integer!"

    if balancing is None or balancing == "NONE":
        dtypes = const.RAW_COO_CSV_DTYPES
    else:
        dtypes = const.NORM_COO_CSV_DTYPES

    df = load_csv(
        path,
        resolution,
        dtypes=dtypes,
        col_names=const.COO_CSV_NAMES
    )
    
    split_chr1_region = chr1_region.split(':')
    if len(split_chr1_region) == 1:
        #TODO: check if the chromosome name is correct from COO file
        pass
    elif len(split_chr1_region) == 2:
        chr1_name, chr1_reg = split_chr1_region
        #TODO: check if the chromosome name is correct from COO file
        chr1_start, chr1_end = chr1_reg.split('-')
        
        chr1_start = int(chr1_start)
        chr1_start_idx = np.ceil(chr1_start/resolution).astype(int)
        chr1_end = int(chr1_end)
        chr1_end_idx = np.floor(chr1_end/resolution).astype(int)
        
        #? Filter df
        mask = df[const.ROW_IDS_COLNAME] >= chr1_start_idx
        mask &= df[const.ROW_IDS_COLNAME] < chr1_end_idx
        
        mask &= df[const.COL_IDS_COLNAME] >= chr1_start_idx
        mask &= df[const.COL_IDS_COLNAME] < chr1_end_idx
        
        df = df[mask]
    else:
        raise ValueError("Invalid format for chr1_region: {chr1_region}")

    if ret_df:
        return df
    else:
        return (
            df[const.ROW_IDS_COLNAME].to_numpy(),
            df[const.COL_IDS_COLNAME].to_numpy(),
            df[const.COUNTS_COLNAME].to_numpy(),
        )