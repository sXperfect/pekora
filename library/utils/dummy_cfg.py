from omegaconf import OmegaConf, DictConfig
from .optuna import OptunaType

class RunConfig():

    def __init__(
        self,
        run_cfg:dict,
        params_cfg:dict
    ):
        assert isinstance(run_cfg, dict)
        if not isinstance(params_cfg, dict):
            params_cfg = OmegaConf.to_container(
                params_cfg,
                resolve=True
            )

        self.exp = run_cfg
        self.params_cfg = params_cfg

        self.hparams = {}

    def suggest_param(self, key):
        vals = self.params_cfg[key]

        if isinstance(vals, (dict, DictConfig)):
            val_type = OptunaType(vals['type'])
            val = self.exp[key]

            if val_type == OptunaType.INT:
                val = int(val)
                assert vals['low'] < val < vals['high'], f"Invalid value given {vals}: val"
            elif val_type in [OptunaType.FLOAT, OptunaType.LOG]:
                val = float(val)
                assert vals['low'] < val < vals['high'], f"Invalid value given {vals}: val"
            elif val_type in [OptunaType.CATEGORICAL]:
                assert val in vals, f"Invalid value given {vals}: val"
            else:
                pass
        else:
            val = vals

        self.hparams[key] = val

        return val