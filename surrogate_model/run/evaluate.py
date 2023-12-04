import logging
from typing import Callable, List
from argparse import Namespace

import torch.nn as nn
import numpy as np

from run.predict import predict
from utilities.data import MoleculeDataset
from utilities.scaler import StandardScaler


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metric_func: Callable,
                         logger: logging.Logger = None) -> List[float]:
    """
    Evaluates predictions using a metric function.

    :param preds: A list of lists of shape (data_size, num_tasks) with model predictions.
    :param targets: A list of lists of shape (data_size, num_tasks) with targets.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    info = logger.info if logger is not None else print

    if len(preds) == 0:
        return [float('nan')] * num_tasks
    
    preds_dim = [len(x[0]) for x in preds]
    targets = [np.concatenate(x) for x in zip(*targets)]
    targets = [x.reshape([-1, dim]) for x, dim in zip(targets, preds_dim)] #dimensions correlating to preds for the multitask FFNs (global props)

    results = [metric_func(target, pred) for target,pred in zip(targets, preds)]

    return results


def evaluate(model: nn.Module,
             data: MoleculeDataset,
             num_tasks: int,
             metric_func: Callable,
             batch_size: int,
             args: Namespace,
             scaler: StandardScaler = None,
             logger: logging.Logger = None,) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    preds, _ = predict(
        model=model,
        data=data,
        batch_size=batch_size,
        scaler=scaler
    )

    targets = data.targets(ind_props=args.single_mol_tasks)

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        num_tasks=num_tasks,
        metric_func=metric_func,
        logger=logger
    )

    return results
