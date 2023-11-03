import os
import tempfile as tmp
import numpy as np
import pandas as pd
import hydra
from omegaconf.errors import ConfigKeyError
from omegaconf import DictConfig, OmegaConf
import mlflow
from .optuna import OptunaType
from .pareto import is_pareto_efficient, is_pareto_efficient_simple
from ._base_config import BaseConfig

class MLFlowConfig(BaseConfig):
    
    def __init__(
        self,
        exp:mlflow.entities.Run,
        params_cfg:dict
    ):
        """_summary_

        Args:
            exp (mlflow.entities.Run): Run object returned by `mlflow.search_runs` 
            params_cfg (dict): _description_
        """
        
        self.exp = exp
        self.params_cfg = params_cfg
        
        self.hparams = {}
        
    def suggest_param(self, 
        key
    ):
        """Suggest a value based on `mlflow.Run`.

        Args:
            key (str): _description_

        Raises:
            KeyError: _description_

        Returns:
            _type_: _description_
        """
        
        try:
            vals = self.params_cfg[key]
        except (ConfigKeyError, KeyError):
            raise KeyError(f"Key not found:{key}")
        
        #? If vals is optuna optimize style, retrieve value from mlflow.Run
        #? dict is allowed for resolved hydra config
        if isinstance(vals, (dict, DictConfig)):
            val_type = OptunaType(vals['type'])#? Enumerate value
            val = self.exp.data.params[key]

            if val_type == OptunaType.INT:
                val = int(val)
            elif val_type in [OptunaType.FLOAT, OptunaType.LOG]:
                val = float(val)
            #? Do nothing as the value is categorical with all possible type
            elif val_type == OptunaType.CATEGORICAL:
                pass
            else:
                raise RuntimeError(
                    "Should never be reached as the data type is already checked previously"
                )
        else:
            val = vals
                
        self.hparams[key] = val
        
        return val
    
def load_mlf_exp(
    tracking_uri,
    search_experiments_kwargs,
    search_run_args=None,
    pareto_cond:dict=None,
    ret_best_pareto=True,
):
    """_summary_

    Args:
        tracking_uri (_type_): _description_
        search_experiments_kwargs (_type_): _description_
        search_run_args (_type_, optional): _description_. Defaults to None.
        pareto_cond (dict, optional):
            #? TODO: Better explanation
            describing metrics/attributes and if the value must be ascending. Defaults to None.

    Example:
        pareto_cond = {
            metrics.`loss1`: True, #? Means lower is better
            metrics.`loss1`: False, #? Means higher is better
        }

    Returns:
        _type_: _description_
    """
    
    client = mlflow.tracking.MlflowClient(
        tracking_uri=tracking_uri
    )

    found_mlf_exps = client.search_experiments(
        max_results=1,
        **search_experiments_kwargs
    )
     
    assert len(found_mlf_exps), "No experiment found!"

    exp_id = found_mlf_exps[0].experiment_id

    if search_run_args is None:
        search_run_args = dict()
        
    mlf_runs = client.search_runs(
        experiment_ids=exp_id,
        **search_run_args
    )
    
    assert len(mlf_runs), "No experiment found!"
    
    if pareto_cond is not None:
        recs = []
        for ith_mlf_run in mlf_runs:
            rec = dict()
            rec['run_id'] = ith_mlf_run.info.run_id
            rec['mlf_run_obj'] = ith_mlf_run
            for key in pareto_cond.keys():
                key_type, key_name = key.split(".")
                ith_mlf_data = dict(ith_mlf_run.data)
                rec[key] = ith_mlf_data[key_type][key_name.replace('`', '')]
            
            recs.append(rec)
            
        pareto_df = pd.DataFrame.from_records(recs)
        
        for key, is_lower_better in pareto_cond.items():
            if not is_lower_better:
                pareto_df[key] = -1*pareto_df[key]
        
        #? First 2 columns are run_id and mlf_run_obj
        costs = pareto_df.iloc[:, 2:].to_numpy()
        #? Make sure that the values of costs only occupies the 1st quadrant
        costs -= costs.min(axis=0)
        
        if ret_best_pareto:
            pareto_idx = np.argmin(np.prod(costs, axis=1))
            mlf_run = pareto_df.iloc[pareto_idx]["mlf_run_obj"]
                
            return client, mlf_run
        else:
            pareto_mask = is_pareto_efficient(costs)
            mlf_runs = pareto_df.iloc[pareto_mask]["mlf_run_obj"].tolist()
            return client, mlf_runs
        
    #? No pareto front computation
    else:
        mlf_run = mlf_runs[0]
    
    return client, mlf_run

def load_numpy_from_mlf_run(
    client, 
    mlf_run,
    artifact_path,
):
    """Load numpy object from a MLFlow run

    Args:
        client (_type_): _description_
        mlf_run (_type_): _description_
        artifact_path (_type_): a path relative to the root directory of MLflow Runs containing the artifacts to download.

    Returns:
        _type_: _description_
    """
    
    with tmp.TemporaryDirectory() as d:        
        mlflow.artifacts.download_artifacts(
            run_id=mlf_run.info.run_id, 
            tracking_uri=client.tracking_uri,
            artifact_path=artifact_path, 
            dst_path=d
        )
        np_obj = np.load(
            os.path.join(d, artifact_path)
        )
    
    return np_obj