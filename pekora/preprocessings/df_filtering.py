import typing as t
import numpy as np
import pandas as pd
import scipy.sparse as ssparse
from .. import const

def select_val_column(
    df,
    col_name:str=None,
):
    df = df.loc[:, [
        const.ROW_IDS_COLNAME, 
        const.COL_IDS_COLNAME, 
        col_name
    ]]
    
    return df

