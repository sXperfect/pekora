import typing as t
import numpy as np
import pandas as pd
from scipy import stats
from .. import const
from .. import loaders
from .. import utils

def comp_myflamingo_spearmanr(
    chr1_region: str,
    resolution: int,
    balancing: str,
    input_fpath: str,
    points_fpath: str,
    chr2_region: t.Optional[str] = None,
) -> None:
    """
    Compute Spearman rank correlation coefficient between 3C data and distance matrix.

    Parameters
    ----------
    chr1_region : str
        Chromosome region for 3C data.
    resolution : int
        Resolution for 3C data.
    balancing : str
        Balancing method for 3C data.
    input_fpath : str
        Path to input 3C data file.
    points_fpath : str
        Path to points data file.
    chr2_region : Optional[str]
        Chromosome region for second 3C data (optional).

    Returns
    -------
    None

    Notes
    -----
    This function filters out data points that got removed during simulation,
    computes distance matrix from points data, and calculates Spearman rank
    correlation coefficient between 3C data and distance matrix.

    Examples
    --------
    >>> comp_myflamingo_spearmanr('chr1', 10000, ' vanilla', 'input_data.txt', 'points_data.txt')
    Region (Chromosome):chr1; Spearman R:0.800; Data Ratio:0.900

    Authors
    -------
    - Yeremia G. Adhisantoso (adhisant@tnt.uni-hannover.de)
    - Llama3.1 70B - 4.0bpw
    """

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

    #? Create mapping from row/col ids to points
    #? Reason: not all loci are valid, yet the points contains only valid points
    unique_data_ids = np.unique([row_ids, col_ids])

    points_df = pd.read_csv(
        points_fpath,
        sep=',',
        index_col=None,
    )

    points_ids = points_df['frag_id'].to_numpy() - 1

    #? Filter out data that got removed during simulation
    unique_data_ids_mask = np.isin(unique_data_ids, points_ids)
    removed_data_ids = unique_data_ids[~unique_data_ids_mask]
    valid_data_mask = ~np.isin(row_ids, removed_data_ids)
    valid_data_mask &= ~np.isin(col_ids, removed_data_ids)

    row_ids = row_ids[valid_data_mask]
    col_ids = col_ids[valid_data_mask]
    counts = counts[valid_data_mask]

    unique_data_ids = unique_data_ids[unique_data_ids_mask]
    mapping = np.searchsorted(unique_data_ids, np.arange(unique_data_ids.max()+1))

    new_row_ids = mapping[row_ids]
    new_col_ids = mapping[col_ids]

    D = utils.edm_numpy.compute_D_from_Ps(
        points_df[['x', 'y', 'z']].to_numpy(),
        new_row_ids,
        new_col_ids
    )

    res = stats.spearmanr(counts, D)
    corr = res.correlation
    data_ratio = valid_data_mask.sum() / valid_data_mask.size

    print(f"Region (Chromosome):{chr1_region}; Spearman R:{corr:.03f}; Data Ratio:{data_ratio:.03f}")