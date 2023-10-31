from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from .early_stopping import EarlyStopping

CALLBACKS = {
    "EarlyStopping": EarlyStopping,
    "LearningRateMonitor": LearningRateMonitor,
    "ModelCheckpoint": ModelCheckpoint,
}

def init_callbacks(
    cfg:DictConfig, 
    tmp_dir=None
):
    callbacks = []
    if cfg is None:
        return callbacks
    for callback_name, callback_kwargs in cfg.items():
        if callback_name not in CALLBACKS:
            raise ValueError(f"Invalid callback:{callback_name}")
        
        #? By default (if None) give a dictionary
        if callback_kwargs is None:
            callback_kwargs = dict()
        #? Convert Hydra config to dict and list
        elif isinstance(callback_kwargs, DictConfig):
            callback_kwargs = OmegaConf.to_container(callback_kwargs, resolve=True)
        
        if callback_name == 'ModelCheckpoint':            
            callback_kwargs['dirpath'] = tmp_dir
        
        callback = CALLBACKS[callback_name](**callback_kwargs)
        callbacks.append(callback)
        
    return callbacks