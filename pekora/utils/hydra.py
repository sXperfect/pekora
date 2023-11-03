from os.path import basename, join, exists
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

def init_hydra_and_check_config(
    cfg:DictConfig,
    script_name=None,
    check=False
):
    hydra_cfg = HydraConfig.get()
    
    if check:
        exp_selected = hydra_cfg.runtime.choices.exp
        
        if script_name is None:
            script_name = hydra_cfg.job.name
            
        assert cfg.args.python_fname == script_name, \
            f"Invalid config for this script: {exp_selected}"

    for key, path in dict(cfg.path).items():
        if key.startswith('_'):
            continue
        assert exists(path), f"{key} path does not exist: {path}"

    missing_keys = OmegaConf.missing_keys(cfg)
    if len(missing_keys):
        raise RuntimeError(f"The following keys are missing {missing_keys}")
    
    return hydra_cfg