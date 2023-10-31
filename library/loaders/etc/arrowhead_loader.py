import numpy as np
import pandas as pd
from ... import utils

def load_arrowhead(
    fpath
):
    
    df = pd.read_csv(
        fpath, 
        # names=col_names, 
        # dtype=dtypes,
        # **const.CSV_SPEC
    )
    
    pass