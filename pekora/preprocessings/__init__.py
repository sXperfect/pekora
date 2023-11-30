import numpy as np
import pandas as pd
from .. import const
from .wish_dist import comp_sparse_wish_dist, comp_sparse_wish_dist_coo
from .rc_ids import project_row_col_ids
from .df_filtering import (
    select_val_column
)

def preproc_data(
    row_ids,
    col_ids,
    C_vals,
    alpha=-0.25,
    na_inf_val=None,
):

    assert (row_ids <= col_ids).all(), \
        "Input must be the upper triangle of the contact matrix!"

    assert not np.isnan(C_vals).any(), \
        "Input contact matrix must contain no NaN!"

    row_ids, col_ids, C_vals, D_vals = comp_sparse_wish_dist_coo(
        row_ids,
        col_ids,
        C_vals,
        alpha
    )

    mask = np.logical_or(np.isinf(D_vals), np.isnan(D_vals))
    if na_inf_val is None:
        mask = ~mask
        row_ids = row_ids[mask]
        col_ids = col_ids[mask]
        C_vals = C_vals[mask]
        D_vals = D_vals[mask]
    else:
        D_vals = D_vals.copy()
        D_vals[mask] = na_inf_val

    return row_ids, col_ids, C_vals, D_vals

def count_to_dist(
    df:pd.DataFrame,
    alpha=-0.25,
    na_inf_val=None,
):

    assert (df[const.ROW_IDS_COLNAME] <= df[const.COL_IDS_COLNAME]).all(), \
        "Input must be the upper triangle of the contact matrix!"

    assert not np.isnan(df[const.COUNTS_COLNAME]).any(), \
        "Input contact matrix must contain no NaN!"

    df = comp_sparse_wish_dist(
        df,
        alpha=alpha,
        na_inf_val=na_inf_val,
    )

    return df

AVAIL_PREPROC = {
    'comp_sparse_wish_dist': comp_sparse_wish_dist,
    'project_row_col_ids': project_row_col_ids,
    'select_val_column': select_val_column,
}

def preproc_pipeline(
    df:pd.DataFrame,
    configs:dict=None,
):
    if configs is None:
        return df
    else:
        for preproc_name, preproc_cfg in configs.items():
            preproc_f = AVAIL_PREPROC[preproc_name]
            preproc_kwargs = preproc_cfg['kwargs']
            
            if preproc_kwargs is None:
                df = preproc_f(df)
            else:
                df = preproc_f(df, **preproc_kwargs)

        return df