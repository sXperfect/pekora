import os

import numpy as np
import pandas as pd

from .. import const
from .. import loaders

def create_contact_coo(
    chr1_region:str,
    resolution:int,
    balancing:str,
    input:str,
    output:str,
    chr2_region:str=None,
    overwrite:bool=False,
    res_to_one=False,
    norm_pos=False,
    generate_pseudo_weights=False,
    output_delimiter=const.DEF_SEP,
    columns_order:list=None,
):
    assert not os.path.exists(output) or not overwrite, "File exist!"

    df = loaders.load_3c_data(
        input,
        chr1_region,
        resolution,
        balancing=balancing,
        chr2_region=chr2_region,
        ret_df=True
    )
    
    if res_to_one:
        resolution = 1
    else:
        df[[const.ROW_IDS_COLNAME, const.COL_IDS_COLNAME]] *= resolution
        
    if norm_pos:
        min_pos = min(df[const.ROW_IDS_COLNAME].min(), df[const.COL_IDS_COLNAME].min())
        df[[const.ROW_IDS_COLNAME, const.COL_IDS_COLNAME]] -= min_pos - resolution 
        pass
    
    if columns_order is not None:
        for col_name in columns_order:
            assert col_name in df.columns, f"Column {col_name} is not in the data!"
            
        df = df.loc[:, columns_order]
        
    pass
    
    df.to_csv(
        output,
        header=False,
        index=False,
        sep=output_delimiter,
    )

    if generate_pseudo_weights:
        weight_fpath = output.replace(".coo", ".weights")

        max_pos = max(df[const.ROW_IDS_COLNAME].max(), df[const.COL_IDS_COLNAME].max())
        num_weights = np.ceil(max_pos/resolution).astype(int)+1
        df = pd.DataFrame()
        df.insert(0, 'weights', np.ones(num_weights))

        df.to_csv(
            weight_fpath,
            header=False,
            index=False,
            sep=const.DEF_SEP,
        )