import abc
import pandas as pd

class BaseConfig(abc.ABC):
    """
    """
    
    @abc.abstractmethod
    def suggest_param(self, 
        key:str
    ):
        pass
    
    def suggest_dict_params(self, 
        prefix:str
    ):

        keys = pd.Series(self.params_cfg.keys())
        _prefix = prefix + "-"
        mask = keys.str.contains(_prefix, regex=True)

        params = dict()
        for key in keys[mask].values:
            new_key = key.replace(_prefix, "")
            params[new_key] = self.suggest_param(key)
        
        return params