import logging
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data import PredDataset, normalize_data, scaling_back
from ffn_model import ForwardFFN
from args import get_args
from utils import setup_logger


if __name__ == '__main__':

    # initialize
    args = get_args()
    train_dir = args.trained_dir
    save_pred = args.save_dir
    rxn_id_column = args.rxn_id_column

    # logger
    setup_logger(save_pred, log_name="ffn_pred.log", debug=args.debug)
    pl.utilities.seed.seed_everything(args.random_state)

    # Dump args
    yaml_args = yaml.dump(args, indent=2, default_flow_style=False)
    logging.info(f"Args:\n{yaml_args}")
    with open(Path(train_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    df = pd.read_csv(args.pred_file, index_col=0)

    predicted_activation_energies_ind = []

    for i in range(args.ensemble_size):

        scalers_dir_path = Path(train_dir) / f"scalers_{i}"

        scaler = pickle.load(
            open(
                Path(scalers_dir_path) / f"scaler_{i}.pickle",
                "rb",
            )
        )

        # normalize data
        test_scaled, _ = normalize_data(df, rxn_id_column, scalers=scaler)

        # process data
        test_dataset = PredDataset(
            test_scaled,
            args.features,
            train=False
        )

        # Define dataloaders
        test_loader = DataLoader(test_dataset,
                                  shuffle=False,
                                  batch_size=64)


        best_checkpoint = Path(train_dir) / f"best_model_{i}.ckpt"

        # Load from checkpoint
        model = ForwardFFN.load_from_checkpoint(best_checkpoint)
        logging.info(f"Loaded model with from {best_checkpoint}")

        model.eval()
        if args.gpu:
            model = model.cuda()

        idxs, preds = [], []
        with torch.no_grad():
            for batch in test_loader:
                desc = batch
                if args.gpu:
                    desc = desc.cuda()
                output = model(desc)
                preds.append(output)


        pred_values_TS = scaling_back(preds, scaler[args.target_column])

        predicted_activation_energies_ind.append(pred_values_TS)

    # determine ensemble predictions

    predicted_activation_energies = \
        np.sum(predicted_activation_energies_ind, axis=0) / len(predicted_activation_energies_ind)

    final_csv = Path (save_pred) / 'pred.csv'
    predicted_activation_energies = pd.DataFrame(predicted_activation_energies, columns=['DG_TS_tunn'])
    predicted_activation_energies.to_csv(final_csv)
    logging.info(f"Predictions done")
    logging.info(f"Save in {final_csv}")




