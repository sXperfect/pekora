import enum
import dataclasses

class ExperimentType(enum.IntEnum):
    dilution_HIC = 0
    in_situ_HIC = 1
    
class EnzymeName(enum.IntEnum):
    HindIII = 0
    MboI = 1
    NcoI = 2
    DpnII = 3
    MspI = 4
    NcoI_MspI_BspHI = 5
    
class Biosample(enum.IntEnum):
    GM12878 = 0
    HUVEC_cell = 1
    _192627 = 3
    CC_2551 = 4
    CH12_LX = 5
    IMR_90 = 6
    K562 = 7
    KBM_7 = 8
    
USECOLS = [
    "Experiment Set Accession",
    "Experiment Accession",
    "File Accession",
    "Size (MB)",
    "File Type",
    "File Format",
    "Biosource",
    "Organism",
    "Dataset",
    "In Experiment As",
]

TSV_SPEC = dict(
    delimiter="\t",
    comment="#",
)