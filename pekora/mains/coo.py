import os
import typing as t

import numpy as np
import pandas as pd

from .. import const
from .. import loaders

def create_contact_coo_from_args(args):
    
    create_contact_coo(
        chr1_region=args.chr1_region,
        resolution=args.resolution,
        balancing=args.balancing,
        input=args.input,
        output=args.output,
        chr2_region=args.chr2_region,
        overwrite=False,  # Not parsed from argparser
        res_to_one=args.output_resolution_one,
        norm_pos=args.normalize_pos,
        generate_pseudo_weights=args.gen_pseudo_weights,
        output_delimiter=args.output_delimiter,
        columns_order=None  # Not parsed from argparser
    )


def create_contact_coo(
    chr1_region: str,
    resolution: int,
    balancing: str,
    input: str,
    output: str,
    chr2_region: t.Optional[str] = None,
    overwrite: bool = False,
    res_to_one: bool = False,
    norm_pos: bool = False,
    generate_pseudo_weights: bool = False,
    output_delimiter: str = const.DEF_SEP,
    columns_order: t.Optional[t.List[str]] = None,
) -> None:
    """
    Creates a contact matrix in COO format.

    Parameters
    ----------
    chr1_region : str
        Region of the first chromosome.
    resolution : int
        Resolution of the contact matrix.
    balancing : str
        Balancing method used to create the contact matrix.
    input : str
        Path to the input file.
    output : str
        Path to the output file.
    chr2_region : str, optional
        Region of the second chromosome. Defaults to None.
    overwrite : bool, optional
        Whether to overwrite existing output file. Defaults to False.
    res_to_one : bool, optional
        Whether to set resolution to 1. Defaults to False.
    norm_pos : bool, optional
        Whether to normalize positions. Defaults to False.
    generate_pseudo_weights : bool, optional
        Whether to generate pseudo weights. Defaults to False.
    output_delimiter : str, optional
        Delimiter used in the output file. Defaults to const.DEF_SEP.
    columns_order : list of str, optional
        Order of columns in the output file. Defaults to None.

    Returns
    -------
    None
    """
    
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