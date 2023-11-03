import io
import tempfile as temp
from ... import utils
from .. import const
from .coo_loader import load_coo

def load_hic_fast(
    path,
    chr1_region:str,
    resolution:int,
    balancing:str=None,
    in_memory=True,
    chr2_region:str=None,
    ret_df=False,
):    
    if balancing is None:
        balancing = "NONE"
    
    with temp.NamedTemporaryFile() as tmpfile:        
        res = utils.run_straw(
            chr1_region,
            resolution, 
            path, 
            tmpfile.file if not in_memory else None, #? output_f
            balancing=balancing,
            chr2_region=chr2_region
        )
        
        out = load_coo(
            tmpfile.name if not in_memory else io.BytesIO(res),
            chr1_region,
            resolution,
            balancing=balancing,
            ret_df=ret_df,
        )

    return out