import enum
import warnings

class FinalizeStatus(str, enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"
    FINISHED = "finished"
    
def ignore_pl_warnings(
    dataloader_num_workers:bool=True,
    slurm_srun:bool=True,
    mixed_precision:bool=True,
):
    if dataloader_num_workers:
        warnings.filterwarnings("ignore", ".*train_dataloader, does not have many workers.*")
    if slurm_srun:
        warnings.filterwarnings("ignore", ".*The `srun` command is available on your system.*")
    if mixed_precision:
        warnings.filterwarnings("ignore", ".*16 is supported for historical reasons but its usage is discouraged.*")