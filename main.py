import tempfile as tmp
from os import makedirs
from os.path import join

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl

from pekora import const
from pekora import preprocessings
from pekora import loaders
from pekora import utils
from pekora import models
from pekora import archs
from pekora.losses import select_loss
from pekora import datasets as mydatasets

def load_dataset(cfg:DictConfig):
    df = loaders.load_3c_data(
        join(cfg.path.dataset, cfg.args.input),
        cfg.args.chr,
        cfg.args.res,
        balancing=cfg.args.balancing,
        ret_df=True
    )

    return df

def objective_f(
    hydra_cfg,
    trial,
    cfg:DictConfig,
    df:pd.DataFrame=None,
    mode:str='run',
):

    if isinstance(trial, dict):
        trial_cfg = utils.RunConfig(trial, cfg.run_config.parameters)
    else:
        raise ValueError("Unsupported type of variable object")

    alpha = trial_cfg.suggest_param("preproc-basic_preproc-alpha")
    
    #? Create train_df containing "wish distance" also with main diagonal removed
    train_df = preprocessings.basic_preproc_df(
        df.copy() if mode == 'optimize' else df, #? To minimize memory consumption for single run
        alpha=alpha
    )

    if cfg.ds.preproc_pipeline is not None:
        pipeline_configs = utils.optimize_preproc_pipeline(
            trial_cfg,
            dict(**cfg.ds.preproc_pipeline)
        )
    else:
        pipeline_configs = None

    #? Create prerocessed df that includes data filtering
    preproc_df = preprocessings.preproc_pipeline(
        train_df.copy(),
        pipeline_configs
    )

    valid_ids = np.unique(
        np.concatenate(
            [
                preproc_df.loc[:, const.ROW_IDS_COLNAME],
                preproc_df.loc[:, const.COL_IDS_COLNAME]
            ]
        )
    )

    mask = train_df.loc[:, const.ROW_IDS_COLNAME].isin(valid_ids)
    mask &= train_df.loc[:, const.COL_IDS_COLNAME].isin(valid_ids)
    train_df = train_df.loc[mask, :]

    row_ids = preproc_df[const.ROW_IDS_COLNAME].to_numpy()
    col_ids = preproc_df[const.COL_IDS_COLNAME].to_numpy()
    D_vals = preproc_df[const.DIST_COLNAME].to_numpy()

    n = np.amax([np.amax(row_ids), np.amax(col_ids)])+1
    shape = (n, n)

    ###? Extract
    train_dl = DataLoader(
        mydatasets.DummyDataset(cfg.args.num_iters),
        batch_size=1,
        shuffle=False
    )

    lr = trial_cfg.suggest_param("learning_rate")

    optimizer_name = trial_cfg.suggest_param("optimizer")

    if optimizer_name == 'adam':
        optimizer_kwargs = {
            'betas' : [
                trial_cfg.suggest_param("adam-beta1"),
                trial_cfg.suggest_param("adam-beta2")
            ]
        }

    elif optimizer_name == 'sgd':
        optimizer_kwargs = {
            'momentum' : trial_cfg.suggest_param("sgd-momentum"),
            'weight_decay': trial_cfg.suggest_param("sgd-weight_decay"),
        }
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")

    scheduler_name = trial_cfg.suggest_param("scheduler")
    if scheduler_name == 'borzelai-borwein':
        scheduler_kwargs = {
            'lr': lr,
            'max_lr': trial_cfg.suggest_param("bb-max_lr"),
            'min_lr': trial_cfg.suggest_param("bb-min_lr"),
            'steps': trial_cfg.suggest_param("bb-steps"),
            'beta': trial_cfg.suggest_param("bb-beta"),
            'weight_decay':trial_cfg.suggest_param("bb-weight_decay"),
        }
    if scheduler_name == "MultiStepLR":
        scheduler_kwargs = trial_cfg.suggest_dict_params("MultiStepLR")
    else:
        scheduler_kwargs = {}

    arch_obj = archs.SortedEDMArchV1(
        row_ids,
        col_ids,
        None,
        D_vals,
        shape,
        num_batches=cfg.args.num_iters
    )

    reg_Y_name = trial_cfg.suggest_param("model-reg_Y_name")

    if reg_Y_name is not None:
        reg_Y_weight = trial_cfg.suggest_param("model-reg_Y_weight")
        if reg_Y_name == 'ksa':
            reg_Y_kwargs = {
                'order': trial_cfg.suggest_param("model-reg_Y_kwargs-ksa_order"),
                'nonlin': trial_cfg.suggest_param("model-reg_Y_kwargs-nonlin"),
            }
    else:
        reg_Y_kwargs = None
        reg_Y_weight = None
        
    
    reg_P_dist_name = trial_cfg.suggest_param("model-reg_P_dist_name")
    if reg_P_dist_name:
        reg_P_dist_weight = trial_cfg.suggest_param("model-reg_P_dist_weight")
        
        reg_P_dist_kwargs = {
            "unique_point_ids": valid_ids, #? Mandatory!
            "mode": trial_cfg.suggest_param("model-reg_P_dist-mode")
        }
    else:
        reg_P_dist_kwargs = None
        reg_P_dist_weight = None

    model_hparams = dict(
        loss=trial_cfg.suggest_param("model-loss"),
        reg_Y=reg_Y_name,
        reg_Y_kwargs=reg_Y_kwargs,
        reg_Y_weight=reg_Y_weight,
        reg_P_dist=reg_P_dist_name,
        reg_P_dist_kwargs=reg_P_dist_kwargs,
        reg_P_dist_weight=reg_P_dist_weight,
    )    

    model_obj = models.RegularizedGradientDescent(
        arch_obj,
        lr=lr,
        alpha=alpha,
        optimizer=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        scheduler=scheduler_name,
        scheduler_kwargs=scheduler_kwargs,
        **model_hparams
    )

    torch.set_float32_matmul_precision('highest')

    if mode == "optimize":
        trial_name = f"{trial.number:04d}"
    else:
        trial_name = f"res{cfg.args.res}-chr{cfg.args.chr}-balancing_{cfg.args.balancing}"

    if cfg.args.dry_run is True or cfg.pl.logger is None:
        logger = None
    else:
        tags = dict(**cfg.ds.spec)
        tags['accelerator'] = cfg.args.accelerator
        tags['precision'] = cfg.args.precision
        
        for k, v in tags.items():
            tags[k] = str(v)
        
        logger = utils.logger.init_logger(
            cfg.pl.logger,
            trial_name=trial_name,
            tags=tags,
        )

    #? Temporary Directory must be created before callbacks are created
    tmp_dir = tmp.TemporaryDirectory()

    callbacks = utils.callback.init_callbacks(
        cfg.pl.callbacks,
        tmp_dir=tmp_dir.name
    )

    trainer = pl.Trainer(
        default_root_dir=tmp_dir.name,
        logger=logger,
        callbacks=callbacks,
        **cfg.pl.trainer
    )

    try:
        trainer.fit(model_obj, train_dl)
        
        if cfg.args.dry_run is False: 
            P = model_obj.arch.P.detach().numpy()
            fname = f"{hydra_cfg.runtime.choices.exp}-{cfg.args.input}-{cfg.args.chr}-{cfg.args.res}.npy"
            np.save(
                join(
                    cfg.path.output,
                    fname
                ),
                P
            )

        tmp_dir.cleanup()

    except Exception as e:
        if logger is not None:
            logger.finalize(utils.FinalizeStatus.FAILED)

        tmp_dir.cleanup()
        raise RuntimeError(str(e))

           
    if mode == 'run':
        pass
        D = (
            model_obj
            .compute_D_nograd(
                train_df[const.ROW_IDS_COLNAME].to_numpy(), #? It does not contains the main diag
                train_df[const.COL_IDS_COLNAME].to_numpy()  #? It does not contains the main diag
            )
            .detach()
            .numpy()
        )
        
        # mse_loss_f = select_loss('mse_np')
        # mse_loss = mse_loss_f(
        #     D, 
        #     train_df[const.DIST_COLNAME].to_numpy() #? It does not contains the main diag
        # )
        
        #? Negative value because compared against D and the correlation is inverse
        score = (
            -stats.spearmanr(
                D,
                train_df[const.COUNTS_COLNAME].to_numpy()
            )
            .statistic
        )
        
        print(f"Spearman's rank correlation: {score}")

        # metrics = {
        #     "metric-mse": mse_loss, #? MSE of all data
        #     "metric-spearmann_r": score #? Spearman of all data
        # }
        
        # if cfg.args.dry_run is False and logger is not None:  
        #     utils.logger.log_hparams_metrics(
        #         logger,
        #         metrics=metrics
        #     )
    else:
        raise ValueError(f"Invalid mode:{mode}")

@hydra.main(
    version_base=None, 
    config_path="configs", 
    config_name="base_config"
)
def main(cfg:DictConfig):
    ###! DO NOT CHANGE ###
    utils.ignore_pl_warnings()
    hydra_cfg = utils.init_hydra_and_check_config(cfg)

    missing_keys = OmegaConf.missing_keys(cfg)
    if len(missing_keys):
        raise RuntimeError(f"The following keys are missing {missing_keys}")

    mode = cfg.args.mode.lower()
    ###! DO NOT CHANGE ###

    df = load_dataset(cfg)

    if mode == 'run':
        run_cfg = OmegaConf.to_container(
            cfg.run_config.parameters,
            resolve=True
        )

        objective_f(
            hydra_cfg,
            run_cfg,
            cfg,
            df=df,
            mode=mode,
        )
    else:
        raise NotImplementedError(f"Invalid mode:{mode}")

if __name__ == "__main__":
    main()
