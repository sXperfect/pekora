import os

import numpy as np
import pandas as pd

from .. import const
from .. import loaders

def create_contact_mcoo(
    chr1_region:str,
    resolution:int,
    balancing:str,
    input:str,
    output:str,
    chr2_region:str=None,
    overwrite:bool=False,
):

    assert not os.path.exists(output) or not overwrite, "File exist!"

    norm_count_df = (
        loaders.load_3c_data(
            input,
            chr1_region,
            resolution,
            balancing=balancing,
            chr2_region=chr2_region,
            ret_df=True
        )
        .rename(
            columns={const.COUNTS_COLNAME: const.NORM_COUNTS_COLNAME}
        )
    )

    raw_count_df = (
        loaders.load_3c_data(
            input,
            chr1_region,
            resolution,
            balancing=None,
            chr2_region=chr2_region,
            ret_df=True
        )
        .rename(
            columns={const.COUNTS_COLNAME: const.RAW_COUNTS_COLNAME}
        )
    )

    df = (
        norm_count_df
        .merge(
            raw_count_df, 
            how='left'
        )
    )

    assert not df.isna().any().any(), \
        "Found NaN value!"
    assert not (df[const.ROW_IDS_COLNAME] > df[const.COL_IDS_COLNAME]).any(), \
        "Found an entry in lower triangle of the matrix!"

    df[[const.ROW_IDS_COLNAME, const.COL_IDS_COLNAME]] *= resolution
    df.to_csv(
        output,
        header=False,
        index=False,
        sep=const.DEF_SEP,
    )