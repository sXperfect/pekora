
import hydra
from omegaconf import DictConfig, OmegaConf
from ml_library.utils import (
    ignore_pl_warnings, 
    init_hydra_and_check_config
)

@hydra.main(
    version_base=None, 
    config_path="../configs", 
    config_name="base_exp"
)
def main(cfg:DictConfig):

    ignore_pl_warnings()
    hydra_cfg = init_hydra_and_check_config(cfg)

    missing_keys = OmegaConf.missing_keys(cfg)
    if len(missing_keys):
        raise RuntimeError(f"The following keys are missing {missing_keys}")

    mode = cfg.args.mode
    
    return cfg, hydra_cfg, mode