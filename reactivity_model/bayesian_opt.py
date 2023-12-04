import torch

from data import normalize_data, scaling_back, PredDataset, PredDataset_fps
from data import get_metrics
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from ffn_model import ForwardFFN

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from utils import setup_logger, ConsoleLogger
import logging

from pathlib import Path
import yaml
import argparse

from types import SimpleNamespace

from functools import partial
from hyperopt import fmin, hp, tpe

import os

from data import get_fingerprints_all_rxn


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default=None, type=str, help="Path to data"
    )
    parser.add_argument(
        "--model_dir",
        default="results/bayesian_opt",
        help="path to the checkpoint file of the trained model",
    )
    parser.add_argument(
        '--features',
        nargs="+",
        type=str,
        default=['dG_forward', 'dG_reverse', 'q_reac0', 'qH_reac0', 'q_reac1', 's_reac1', 'q_prod0',
                 's_prod0', 'q_prod1', 'qH_prod1', 'BV_reac1', 'BV_prod0', 'fr_dG_forward', 'fr_dG_reverse'],
        help='features for the different models')
    parser.add_argument(
        '--csv_file',
        type=str,
        default=None,
        help='path to file containing the rxn-smiles',
    )
    parser.add_argument(
        "--random_state",
        default=1,
        action="store",
        type=int,
        help="random state to be selected for sampling/shuffling")

    # interactive way
    parser.add_argument("--mode", default='client', action="store", type=str)
    parser.add_argument("--host", default='127.0.0.1', action="store", type=str)
    parser.add_argument("--port", default=57546, action="store", type=int)

    cl_args = parser.parse_args()
    return cl_args


def objective(args0,
    df_reactions,
    rxn_id_column,
    target_column,
    model_dir,
    k_fold,
    selec_batch_size,
    features,
    random_state,
    ):

    args = SimpleNamespace(**args0)

    pl.utilities.seed.seed_everything(random_state)

    # split df into k_fold groups
    k_fold_arange = np.linspace(0, len(df_reactions), k_fold + 1).astype(int)

    # create lists to store metrics for each fold
    rmse_activation_energy_list = []

    for i in range(k_fold):

        # make a directory to store model files
        fold_dir = Path(model_dir) / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # split data for fold
        if df_reactions is not None:
            valid = df_reactions[k_fold_arange[i]: k_fold_arange[i + 1]]
            train = df_reactions[
                ~df_reactions[f"{rxn_id_column}"].isin(
                    valid[f"{rxn_id_column}"]
                )
            ]

        # normalize data
        train_scaled, train_scalers = normalize_data(train, rxn_id_column)
        valid_scaled, _ = normalize_data(valid, rxn_id_column, scalers=train_scalers)

        # process data
        train_dataset = PredDataset(
            train_scaled,
            features,
            target_column,
        )
        valid_dataset = PredDataset(
            valid_scaled,
            features,
            target_column,
        )

        # Define dataloaders
        train_loader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=selec_batch_size)
        val_loader = DataLoader(valid_dataset,
                                shuffle=False,
                                batch_size=selec_batch_size)

        # Define model
        test_batch = next(iter(train_loader))

        input_dim = train_dataset.getdimension()
        model = ForwardFFN(
            hidden_size=int(args.hidden_size),
            layers=int(args.layers),
            dropout=0.0,
            learning_rate=float(args.learning_rate),
            min_lr=0.01,
            input_dim=input_dim,
            output_dim=1,
            lr_ratio=args.lr_ratio
        )

        # Create trainer
        tb_logger = pl_loggers.TensorBoardLogger(fold_dir, name="", version="")
        console_logger = ConsoleLogger()

        tb_path = tb_logger.log_dir
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=tb_path,
            filename=f"best_model_{i}",
            save_weights_only=False,
        )
        earlystop_callback = EarlyStopping(monitor="val_loss", patience=10)
        callbacks = [earlystop_callback, checkpoint_callback]

        trainer = pl.Trainer(logger=tb_logger,
                             accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             gpus=1 if torch.cuda.is_available() else 0,
                             callbacks=callbacks,
                             gradient_clip_val=5,
                             max_epochs=100,
                             gradient_clip_algorithm="value",
                             )

        trainer.fit(model, train_loader, val_loader)
        checkpoint_callback = trainer.checkpoint_callback
        best_checkpoint = checkpoint_callback.best_model_path
        best_checkpoint_score = checkpoint_callback.best_model_score.item()

        # Load from checkpoint
        model = ForwardFFN.load_from_checkpoint(best_checkpoint)

        model.eval()
        valid_pred = trainer.predict(model=model, dataloaders=val_loader)

        pred_values_TS = scaling_back(valid_pred, train_scalers[target_column])

        # compute and store metrics
        mae_act, rmse_act = get_metrics(valid[target_column], pred_values_TS)

        rmse_activation_energy_list.append(rmse_act)

    logging.info(f"Selected Hyperparameters: {args0}")
    logging.info(
        f"RMSE amounts to - activation energy: {np.mean(np.array(rmse_activation_energy_list))}\n"
    )

    return np.mean(np.array(rmse_activation_energy_list))


def objective_fps(args0,
    df_reactions,
    rxn_id_column,
    target_column,
    model_dir,
    k_fold,
    selec_batch_size,
    features,
    random_state,
    ):

    features = None

    args = SimpleNamespace(**args0)

    pl.utilities.seed.seed_everything(random_state)

    # split df into k_fold groups
    k_fold_arange = np.linspace(0, len(df_reactions), k_fold + 1).astype(int)

    # create lists to store metrics for each fold
    rmse_activation_energy_list = []

    for i in range(k_fold):

        # make a directory to store model files
        fold_dir = Path(model_dir) / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # split data for fold
        if df_reactions is not None:
            valid = df_reactions[k_fold_arange[i]: k_fold_arange[i + 1]]
            train = df_reactions[
                ~df_reactions[f"{rxn_id_column}"].isin(
                    valid[f"{rxn_id_column}"]
                )
            ]

        # normalize data
        train_scaled, train_scalers = normalize_data(train, rxn_id_column, fps=True)
        valid_scaled, _ = normalize_data(valid, rxn_id_column, fps=True, scalers=train_scalers)

        # process data
        train_dataset = PredDataset_fps(
            train_scaled,
            target_column,
        )
        valid_dataset = PredDataset_fps(
            valid_scaled,
            target_column,
        )

        # Define dataloaders
        train_loader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=selec_batch_size)
        val_loader = DataLoader(valid_dataset,
                                shuffle=False,
                                batch_size=selec_batch_size)

        # Define model
        test_batch = next(iter(train_loader))

        input_dim = train_dataset.getdimension()
        model = ForwardFFN(
            hidden_size=int(args.hidden_size),
            layers=int(args.layers),
            dropout=0.0,
            learning_rate=float(args.learning_rate),
            min_lr=0.01,
            input_dim=input_dim,
            output_dim=1,
            lr_ratio=float(args.lr_ratio)
        )

        # Create trainer
        tb_logger = pl_loggers.TensorBoardLogger(fold_dir, name="", version="")
        console_logger = ConsoleLogger()

        tb_path = tb_logger.log_dir
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=tb_path,
            filename=f"best_model_{i}",
            save_weights_only=False,
        )
        earlystop_callback = EarlyStopping(monitor="val_loss", patience=10)
        callbacks = [earlystop_callback, checkpoint_callback]

        trainer = pl.Trainer(logger=tb_logger,
                             accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             gpus=1 if torch.cuda.is_available() else 0,
                             callbacks=callbacks,
                             gradient_clip_val=5,
                             max_epochs=100,
                             gradient_clip_algorithm="value",
                             )

        trainer.fit(model, train_loader, val_loader)
        checkpoint_callback = trainer.checkpoint_callback
        best_checkpoint = checkpoint_callback.best_model_path
        best_checkpoint_score = checkpoint_callback.best_model_score.item()

        # Load from checkpoint
        model = ForwardFFN.load_from_checkpoint(best_checkpoint)

        model.eval()
        valid_out = trainer.test(model=model, dataloaders=val_loader)
        valid_pred = trainer.predict(model=model, dataloaders=val_loader)

        pred_values_TS = scaling_back(valid_pred, train_scalers[target_column])

        # compute and store metrics
        mae_act, rmse_act = get_metrics(valid[target_column], pred_values_TS)

        rmse_activation_energy_list.append(rmse_act)

    logging.info(f"Selected Hyperparameters: {args0}")
    logging.info(
        f"RMSE amounts to - activation energy: {np.mean(np.array(rmse_activation_energy_list))}\n"
    )

    return np.mean(np.array(rmse_activation_energy_list))



def ffnn_bayesian(
    data_path,
    csv_file,
    random_state,
    model_dir,
    features, ):

    if data_path is not None:
        df = pd.read_pickle(data_path)
    elif csv_file is not None:
        df = pd.read_csv(csv_file, index_col=0)
        df = get_fingerprints_all_rxn(df)

    df = df.sample(frac=0.8, random_state=random_state)

    space = {
        "layers": hp.quniform("layers", low=0, high=3, q=1),
        "hidden_size": hp.quniform("hidden_size", low=10, high=300, q=10),
        "learning_rate": 0.08 - hp.loguniform("learning_rate", low=np.log(0.01), high=np.log(0.08)),
        "lr_ratio": hp.quniform("lr_ratio", low=0.9, high=0.99, q=0.01),
    }

    fmin_objective = partial(
        objective_fps,
        df_reactions=df,
        rxn_id_column="rxn_id",
        target_column="DG_TS_tunn",
        model_dir=model_dir,
        k_fold=4,
        selec_batch_size=10,
        features=features,
        random_state=random_state
    )

    best = fmin(fmin_objective, space, algo=tpe.suggest, max_evals=256)
    logging.info(best)


if __name__ == "__main__":

    cl_args = parse_command_line_args()
    if not os.path.isdir(cl_args.model_dir):
        os.mkdir(cl_args.model_dir)

    # logger
    setup_logger(cl_args.model_dir, log_name="ffn_train.log")

    ffnn_bayesian(
        cl_args.data_path,
        cl_args.csv_file,
        cl_args.random_state,
        cl_args.model_dir,
        cl_args.features,
    )

