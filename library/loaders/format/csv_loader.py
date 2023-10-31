import pandas as pd
from ... import const
  
def load_csv(
    obj, 
    resolution,
    dtypes=None,
    col_names=None,
):
        
    df = pd.read_csv(
        obj, 
        names=col_names, 
        dtype=dtypes,
        **const.CSV_SPEC
    )
    
    df[const.ROW_IDS_COLNAME] = df[const.ROW_IDS_COLNAME] // resolution
    df[const.COL_IDS_COLNAME] = df[const.COL_IDS_COLNAME] // resolution
    
    return df