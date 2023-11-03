import typing as t
import enum
import copy
# import optuna
from omegaconf import DictConfig, OmegaConf
from ._base_config import BaseConfig

class OptunaType(enum.IntEnum):
    r"""
    Different ways optuna_suggest_X operations are used in this codebase
    CAREFUL: This enum duplicated data from configs/base_exp.yaml. Do not change one without the other
    See Also
    --------
    optuna_suggest : Uses this to choose appropriate optuna_suggest_X(...) call
    OptunaConfig
    """

    INT = 0
    FLOAT = 1
    LOG = 2
    CATEGORICAL = 3

# def optuna_suggest(
#     obj:t.Union[optuna.Trial, t.Dict], 
#     type_tag:OptunaType, 
#     name:str, 
#     low:t.Union[int, float]=None, 
#     high:t.Union[int, float]=None, 
#     step:t.Union[int, float]=None, 
#     vals:t.Sequence[t.Any]=None, 
#     **kwargs
# ):
#     """_summary_

#     Args:
#         obj (t.Union[optuna.Trial, t.Dict]):
#             Object to query value from.
#             If `name` is specified in the hydra config as a constant value, obj will be of type dict. If name instead contains at least a dictionary with the at least the key 'type', it will be treated as a Trial.
#         type_tag (OptunaType):
#             Controls which optuna_suggest_X to dispatch to and with which arguments.
#         name (str):
#             Dict Key or Optuna name
#         low (t.Union[int, float], optional): _description_. Defaults to None.
#         high (t.Union[int, float], optional): _description_. Defaults to None.
#         step (t.Union[int, float], optional): _description_. Defaults to None.
#         vals (t.Sequence[t.Any], optional):
#             Choices for categorical values. Defaults to None.

#     Raises:
#         ValueError: _description_

#     Returns:
#         Suggested value from Optuna or a constant value from config file.
#     """

#     if isinstance(obj, optuna.Trial):
#         if type_tag == OptunaType.INT:
#             if step is None:
#                 step = 1
#             return obj.suggest_int(name, low, high, step=step)
#         elif type_tag == OptunaType.FLOAT:
#             return obj.suggest_float(name, low, high, step=step)
#         elif type_tag == OptunaType.LOG:
#             return obj.suggest_float(name, low, high, step=step, log=True)
#         elif type_tag == OptunaType.CATEGORICAL:
#             return obj.suggest_categorical(name, vals)
#         else:
#             raise ValueError("Invalid type!")
        
#     elif isinstance(obj, dict):
#         return obj[name]
    
# class OptunaConfig(BaseConfig):
#     """This class is a wrapper for optuna Trial object.
#     It logs hyperparameter values inside a dictionary.
#     """
    
#     def __init__(
#         self,
#         trial:optuna.Trial,
#         params_cfg:dict
#     ):
#         self.trial = trial
#         self.params_cfg = params_cfg
        
#         self.hparams = {} #? Dictionary for hyperparameter values

#     def suggest_param(
#         self, 
#         key:str
#     ):
#         """
#         Get optuna_suggest_X output for param indexed by key in config. The right optuna_suggest call is decided via other config values

#         Parameters
#         ----------
#         key : Name of the param in the config

#         Returns
#         -------
#         Result of optuna_suggest_X called with the configured arguments

#         """
#         try:  
#             vals = self.params_cfg[key]
#         except:
#             raise KeyError(f'There is no parameter "{key}" in optuna parameters')
        
#         if type(vals) in [DictConfig]:
#             val_type = OptunaType(vals['type'])
#             kwargs = dict(vals)
#             del kwargs['type']
            
#             val = optuna_suggest(self.trial, val_type, name=key, **kwargs)
#         else:
#             val = vals
                
#         #? Log Hyperparameters in a dictionary
#         self.hparams[key] = val
        
#         return val

    
# def optimize_preproc_pipeline(
#     opt_cfg:OptunaConfig,
#     pipeline_configs:t.Dict
# ):
#     """
#     Preprocessing hyperparameters are processed differently than other hyperparams. They appear once in the ds dict, and the ones that get optimized with optuna get tagged with the value '_OPTIMIZE_'. optuna.parameters contains the tagged value with the actual choices to optimize over.

#     Parameters
#     ----------
#     opt_cfg : OptunaConfig
#     pipeline_configs: Dict : ds.preproc_pipeline subconfig from hydra config. Will be changed in place.

#     Returns
#     -------
#     Updated `pipeline_configs`
#     """

#     #? Avoid replacing the original config by creating a copy
#     if isinstance(pipeline_configs, dict):
#         pipeline_configs = copy.deepcopy(pipeline_configs)
#     else:
#         pipeline_configs = OmegaConf.to_container(pipeline_configs, resolve=True)

#     for preproc_name, preproc_vals in pipeline_configs.items():
#         for __, k_args_vals in preproc_vals.items():
#             if k_args_vals is not None:
#                 for param_name, param_val in k_args_vals.items():
#                     if param_val == '_OPTIMIZE_':
#                         optuna_key = f"preproc-{preproc_name}-{param_name}"
#                         suggested_val = opt_cfg.suggest_param(optuna_key)
#                         k_args_vals[param_name] = suggested_val

#     return pipeline_configs
