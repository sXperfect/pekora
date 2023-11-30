import io
import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from scipy.optimize import minimize
import Bio
import Bio.PDB
import Bio.SeqRecord
from .. import const
from .. import loaders
from .. import utils

DELIMITER = "\t"
PDB_COL_NAMES = [
    "ATOM1",
    "ATOM2",
    "ATOM3",
    "ATOM4",
    "ATOM_SERIAL_NUMBER",
    "X", "Y", "Z",
    "A", "B"
]
PDB_USE_COLS = [
    # "ATOM_SERIAL_NUMBER",
    "X", "Y", "Z",
]

PATTERN_F = re.compile(
    r"(-?\d{1,3}.\d{3})\s*(-?\d{1,3}.\d{3})\s*(-?\d{1,3}.\d{3})"
)

def comp_h3dg_spearmanr(
    chr1_region:str,
    resolution:int,
    balancing:str,
    input_fpath:str,
    points_fpath:str,
    mappings_fpath:str,
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

    #? Create mapping from row/col ids to points
    #? Reason: not all loci are valid, yet the points contains only valid points
    unique_data_ids = np.unique([row_ids, col_ids])

    #? Read points using PDB library
    # pdbparser = Bio.PDB.PDBParser(QUIET=True)
    # pbd_obj = pdbparser.get_structure(0, points_fpath)

    #? Read points first style
    #TODO: PBD Bug? number of points is limited to 9,999
    # residues = pbd_obj[0].child_list[0]
    # num_points = len(residues.child_list)
    # points = np.empty((num_points, 3))
    # for i, residue in enumerate(residues):
    #     coor = residue.child_list[0].get_vector()._ar
    #     points[i, :] = coor

    #? Read points second style
    #TODO: PBD Bug? number of points is limited to 9,999
    # points = []
    # for model in pbd_obj:
    #     for chain in model:
    #         for residue in chain:
    #             for atom in residue:
    #                 coor = atom.coord
    #                 points.append(coor)

    points_lines = []
    with open(points_fpath) as f:
        for line in f:
            if line.startswith('ATOM'):
                # points_lines.append(line)
                #? To fix strange lines where there is no delimiter between values
                points_lines.append(
                    "\t".join(PATTERN_F.search(line).groups())
                )

    f = io.StringIO("\n".join(points_lines))
    
    points_df = pd.read_csv(
        f,
        # sep=r"\s+", #? Delimiter is all possible
        delim_whitespace=True, #? Delimiter is all possible whitespace
        header=None,
        names=PDB_USE_COLS,
        # names=PDB_COL_NAMES,
        # usecols=PDB_USE_COLS,
        index_col=None,
    )
    points = points_df.to_numpy()
    
    #? This is the mapping of points to loci (result of simulation)
    mapping_df = pd.read_csv(
        mappings_fpath,
        sep="\t",
        header=None,
        names=['loci', 'id'],
        index_col=None,
    ).to_numpy()

    mapping_df[:, 0] //= resolution

    #? Filter out data that got removed during simulation
    unique_data_ids_mask = np.isin(unique_data_ids, mapping_df[:, 0])
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
    # new_counts = mapping[counts]

    D = utils.edm_numpy.compute_D_from_Ps(
        points,
        new_row_ids,
        new_col_ids
    )
    
    # def alpha_estimate(alpha):
    #     return metrics.mean_squared_error(D, counts**alpha) 

    res = stats.spearmanr(counts, D)
    corr = res.correlation
    data_ratio = valid_data_mask.sum() / valid_data_mask.size

    print(f"Region (Chromosome):{chr1_region}; Spearman R:{corr:.03f}; Data Ratio:{data_ratio:.03f}")