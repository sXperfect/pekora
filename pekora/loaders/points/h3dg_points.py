import io
import numpy as np
import pandas as pd

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
    "X", "Y", "Z",
]

def load_h3dg_points(
    points_fpath, 
    mapping_fpath, 
    resolution
):
    """_summary_

    Args:
        points_fpath (_type_): _description_
        mapping_fpath (_type_): _description_
        resolution (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    #? Read point coordinates from .pdb file
    points_lines = []
    with open(points_fpath) as f:
        for line in f:
            if line.startswith('ATOM'):
                points_lines.append(line)

    #? Create points in numpy format
    f = io.StringIO("".join(points_lines))
    h3dg_P = (
        pd.read_csv(
            f,
            delim_whitespace=True, #? Delimiter is all possible whitespace
            header=None,
            names=PDB_COL_NAMES,
            usecols=PDB_USE_COLS,
            index_col=None,
        )
        .to_numpy()
    )
    
    #? The points loaded contains only loci that are valid for optimization
    #? In order to load the coordinates of all loci, inverse mapping have to be done
    dest_src_ids_mapping = pd.read_csv(
        mapping_fpath,
        sep="\t",
        header=None,
        names=['loci', 'id'],
        index_col=None,
    ).to_numpy()

    dest_src_ids_mapping[:, 0] //= resolution
    
    #? Create canonical point representation based on the mapping
    canonical_P = np.full((dest_src_ids_mapping.max()+1, 3), np.nan)
    canonical_P[dest_src_ids_mapping[:, 0], :] = h3dg_P[dest_src_ids_mapping[:, 1], :]
    
    
    return canonical_P