import torch

from data import split_data_training, normalize_data, PredDataset
import pandas as pd
from torch.utils.data import DataLoader
from ffn_model import ForwardFFN

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from args import get_args
from utils import setup_logger, ConsoleLogger
import logging

from pathlib import Path
import yaml
import pickle

if __name__ == '__main__':

    # initialize
    args = get_args()
    reactivity_data = pd.read_pickle(args.data_path)
    save_dir = args.save_dir
    rxn_id_column = args.rxn_id_column

    # logger
    setup_logger(save_dir, log_name="ffn_train.log", debug=args.debug)
    pl.utilities.seed.seed_everything(args.random_state)

    # Dump args
    yaml_args = yaml.dump(args, indent=2, default_flow_style=False)
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    logging.info(f"Args:\n{yaml_args}")

    for i in range(args.ensemble_size):
        logging.info(f"Training of model {i} started...")
        scalers_dir_path = Path(save_dir) / f"scalers_{i}"
        scalers_dir_path.mkdir(parents=True, exist_ok=True)

        # split data
        train, valid, _ = split_data_training(reactivity_data, rxn_id_column, [0.9, 0.1, 0.0], args.random_state, i)

        # normalize data
        train_scaled, train_scalers = normalize_data(train, rxn_id_column)
        valid_scaled, _ = normalize_data(valid, rxn_id_column, scalers=train_scalers)

        pickle.dump(
            train_scalers,
            open(
                Path(scalers_dir_path) / f"scaler_{i}.pickle",
                "wb",
            ),
        )

        # process data
        train_dataset = PredDataset(
            train_scaled,
            args.features,
            args.target_column,
        )
        valid_dataset = PredDataset(
            valid_scaled,
            args.features,
            args.target_column,
        )

        dataset_sizes = (len(train_dataset), len(valid_dataset))
        logging.info(f"Train, val: {dataset_sizes}")

        # Define dataloaders
        train_loader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=args.batch_size)
        val_loader = DataLoader(valid_dataset,
                                shuffle=False,
                                batch_size=args.batch_size)

        # Define model
        test_batch = next(iter(train_loader))

        input_dim = train_loader.dataset.descriptors.shape[-1]

        if args.transfer_learning:
            trained_checkpoint = Path(args.trained_dir) / f"best_model_{i}.ckpt"
            model = ForwardFFN.load_from_checkpoint(trained_checkpoint)
            # for name, param in model.named_parameters():
            #    if param.requires_grad and 'all_layers.1.0.' in name:
            #        continue
            #    param.requires_grad = False
        else:
            model = ForwardFFN(
                hidden_size=args.hidden_size,
                layers=args.layers,
                dropout=args.dropout,
                input_dim=input_dim,
                output_dim=1,
                learning_rate=args.learning_rate,
                lr_ratio=args.lr_ratio
            )

        # Create trainer
        tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="", version="")
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
                             accelerator="gpu" if args.gpu else "cpu",
                             gpus=1 if args.gpu else 0,
                             callbacks=callbacks,
                             gradient_clip_val=5,
                             max_epochs=args.max_epochs,
                             gradient_clip_algorithm="value")

        trainer.fit(model, train_loader, val_loader)
        checkpoint_callback = trainer.checkpoint_callback
        best_checkpoint = checkpoint_callback.best_model_path
        best_checkpoint_score = checkpoint_callback.best_model_score.item()
    logging.info(f"Training finished")




