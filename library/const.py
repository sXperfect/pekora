import logging as log
import enum

import numpy as np

ROW_IDS_COLNAME = "row_ids"
COL_IDS_COLNAME = "col_ids"
COUNTS_COLNAME = "counts"
RAW_COUNTS_COLNAME = "raw_counts"
NORM_COUNTS_COLNAME = "norm_counts"
DIST_COLNAME = 'dists'
DEF_SEP = '\t'
RAW_COUNTS_DTYPE = np.uint32
NORM_COUNTS_DTYPE = np.float64

CSV_DTYPE = {
    ROW_IDS_COLNAME:np.int64, 
    COL_IDS_COLNAME:np.int64,
}

CSV_SPEC = dict(
    sep=DEF_SEP, 
    index_col=None,
    engine='c'
)


COO_CSV_NAMES = [ROW_IDS_COLNAME, COL_IDS_COLNAME, COUNTS_COLNAME]
MCOO_CSV_NAMES = [ROW_IDS_COLNAME, COL_IDS_COLNAME, COUNTS_COLNAME, RAW_COUNTS_COLNAME]

RAW_COO_CSV_DTYPES = {
    **CSV_DTYPE,
    COUNTS_COLNAME: RAW_COUNTS_DTYPE,
}

NORM_COO_CSV_DTYPES = {
    **CSV_DTYPE,
    COUNTS_COLNAME: NORM_COUNTS_DTYPE,
}

MCOO_CSV_DTYPES = {
    **CSV_DTYPE,
    COUNTS_COLNAME: NORM_COUNTS_DTYPE,
    RAW_COUNTS_COLNAME: RAW_COUNTS_DTYPE,
}

LOG_LEVELS = {
    'critical': log.CRITICAL,
    'error': log.ERROR,
    'warning': log.WARNING,
    'info': log.INFO,
    'debug': log.DEBUG,
}

class Counts(enum.Enum):
    OBSERVED = 'observed'
    OE = 'oe'
    EXPECTED = 'expected'
    
AVAIL_COUNTS = [e.value for e in Counts]
    
class Balancing(enum.Enum):
    NONE = 'NONE'
    KR = 'KR'
    VC = 'VC'
    VC_SQRT = 'VC_SQRT'
    
AVAIL_BALANCINGS = [e.value for e in Balancing]
    
class BpFrag(enum.Enum):
    BP = 'BP'
    FRAG = 'FRAG'
    
AVAIL_BPFRAGS = [e.value for e in BpFrag]