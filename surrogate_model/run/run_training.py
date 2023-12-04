from argparse import Namespace
import csv
from logging import Logger
import os
from pprint import pformat
from typing import List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from tqdm import trange
import pickle
from torch.optim.lr_scheduler import ExponentialLR

from run.evaluate import evaluate, evaluate_predictions
from run.predict import predict
from run.train import train
from utilities.scaler import StandardScaler, MinMaxScaler
from utilities.utils_data import get_data, get_task_names, split_data
from utilities.data import MoleculeDataset
from models.model import build_model
from utilities.utils_nn import param_count
from utilities.utils_gen import build_optimizer, build_lr_scheduler, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint


def run_training(args: Namespace, logger: Logger = None, dataset_loaded: MoleculeDataset = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Print args
    debug(pformat(vars(args)))

    # Get data
    debug('Loading data')

    if dataset_loaded == None:
        data = get_data(path=args.data_path, args=args, logger=logger)
    else:
        data = dataset_loaded
    if args.task_names == None:
        args.task_names = get_task_names(path=args.data_path, args=args)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path, logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

    if args.save_smiles_splits:
        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            if args.use_conf_id and args.use_compound_names:
                with open(os.path.join(args.save_dir, name + '_IDs.csv'), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['smiles','chembl_id','conf_id'])
                    for smiles, chembl, conf in zip(dataset.smiles(),dataset.compound_names(),dataset.conf_ids()):
                        writer.writerow([smiles, chembl, conf])
            elif args.use_compound_names and not args.use_conf_id:
                with open(os.path.join(args.save_dir, name + '_IDs.csv'), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['smiles','chembl_id'])
                    for smiles, chembl in zip(dataset.smiles(),dataset.compound_names()):
                        writer.writerow([smiles, chembl])
            else: 
                with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['smiles'])
                    for smiles in dataset.smiles():
                        writer.writerow([smiles])

    if args.features_scaling: # even if this True will be None if additional features is None 
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)
    
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation
    if args.target_scaling:
        debug('Fitting scaler')
        train_targets_ind = train_data.targets(ind_props=True)
        if not args.no_atom_scaling:
            atom_types = train_data.get_atom_types()
        else:
            atom_types = None
        if args.target_scaler_type == 'StandardScaler':
           scaler = StandardScaler(atom_wise=args.atom_wise_scaling).fit(train_targets_ind, 
                    len_ext=len(args.mol_ext_targets), len_int = len(args.mol_int_targets), atom_types = atom_types, mol_multitasks=not args.single_mol_tasks)
        elif args.target_scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler(atom_wise=args.atom_wise_scaling).fit(train_targets_ind, 
                    len_ext=len(args.mol_ext_targets), len_int = len(args.mol_int_targets), atom_types = atom_types, mol_multitasks=not args.single_mol_tasks)
        else:
            scaler=None
        train_targets = train_data.targets(ind_props=args.single_mol_tasks)
        #import pdb; pdb.set_trace()
        scaled_targets = scaler.transform(train_targets, atom_types = atom_types).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Define loss and metric functions
    loss_func = nn.MSELoss(reduction='none')
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_targets = test_data.targets(ind_props=args.single_mol_tasks) # unscaled

    if args.early_stopping:
        epoch_scores = []

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)
        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args, scheduler_name=args.lr_schedule)

        # Run training
        best_score = float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                data=train_data,
                loss_func=loss_func,
                metric_func=metric_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate(
                model=model,
                data=val_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                args=args,
                scaler=scaler,
                logger=logger,
            )

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, n_iter)

            # average validation score (across tasks)
            avg_val_score = np.mean(np.array(val_scores))
            writer.add_scalar(f'Validation {args.metric}', avg_val_score, n_iter)
            debug(f'Validation {args.metric} = {avg_val_score:.6f}')
            
            # Save model checkpoint if improved validation score
            if args.early_stopping:
                if avg_val_score < best_score:
                    epoch_scores = []
                else:
                    epoch_scores.append(avg_val_score)
                    if len(epoch_scores) == (args.patience):
                        info(f'Early stopping at epoch {epoch+1}')
                        break
            best_score, best_epoch = avg_val_score, epoch
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)        

        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch+1}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)
        
        test_preds, test_smiles_batch = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            logger=logger
        )

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
                writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)

        # average validation score (across tasks)
        avg_test_score = np.mean(np.array(test_scores))
        writer.add_scalar(f'Test {args.metric}', avg_test_score, n_iter)
        debug(f'Test {args.metric} = {avg_test_score:.6f}')

        test_preds_all = []
        for x in test_preds:
            if len(x[0]) > 1:
                x = np.hsplit(x, len(x[0]))
                for x_sub in x:
                    test_preds_all.append(x_sub)
            else:
                test_preds_all.append(x)

    return avg_test_score, test_scores, test_preds_all, test_smiles_batch
