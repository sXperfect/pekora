from .. import const
from .format.hic_loader import load_hic
from .format.fast_hic_loader import load_hic_fast
from .format.mcool_loader import load_mcool
from .format.coo_loader import load_coo
from .format.mcoo_loader import load_mcoo
from .points.h3dg_points import load_h3dg_points
from .etc.arrowhead_loader import load_arrowhead

def load_3c_data(
    path:str,
    chr1_region:str,
    resolution:int,
    balancing:str=None,
    use_fast_func=True,
    chr2_region:str=None,
    ret_df=False,
):
    """Load 3C data from a file given resolution, chromosome regions, and balancing method.

    Args:
        path (str): \
            Path to the file.
        chr1_region (str): \
            Region in the first chromosome.
        resolution (int): \
            Resolution
        balancing (str, optional): \
            Balancing method. Defaults to None.
        use_fast_func (bool, optional): \
            Keep intermediate data in memory. Defaults to True.
        chr2_region (str, optional): \
            Region in the second chromosome. \
            If None the region is equal to the first chromosome region. \
            Defaults to None.
        ret_df (bool, optional): \
            Return pandas DataFrame. Defaults to False.

    Raises:
        ValueError: File extension is not supported.

    Returns:
        _type_: List of row_ids, col_ids, counts and (optional) raw counts or DataFrame.
    """
    # TODO: Add function to check if chr1_region is correct/valid
    # TODO: Add step to fix entries in lower triangle of the matrix
    
    if path.endswith('.hic'):
        if use_fast_func:
            out = load_hic_fast(
                path,
                chr1_region,
                resolution,
                balancing,
                chr2_region=chr2_region,
                ret_df=ret_df,
            )
        else:
            out = load_hic(
                path,
                chr1_region,
                resolution,
                balancing,
                chr2_region=chr2_region,
                ret_df=ret_df,
            )
    elif path.endswith('.mcool') or path.endswith('.cool'):
        out = load_mcool(
            path,
            chr1_region,
            resolution,
            balancing,
            chr2_region=chr2_region,
            ret_df=ret_df,
        )
    elif path.endswith('.coo'):
        out = load_coo(
            path,
            chr1_region, #? with mcoo we cannot select the region anymore
            resolution,
            balancing,
            ret_df=ret_df,
        )
    elif path.endswith('.mcoo'):
        out = load_mcoo(
            path,
            chr1_region, #? with mcoo we cannot select the region anymore
            resolution,
            balancing,
            ret_df=ret_df,
        )
    else:
        raise ValueError("File extension not recognized!")

    if ret_df:
        assert (out[const.ROW_IDS_COLNAME] <= out[const.COL_IDS_COLNAME]).all()
    else:
        assert (out[0] <= out[1]).all()

    return out