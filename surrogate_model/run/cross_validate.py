from argparse import Namespace
from logging import Logger
import os
from typing import Tuple
import torch

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from run.run_training import run_training
from utilities.utils_data import get_task_names, get_data
from utilities.utils_gen import makedirs, create_logger
from utilities.data import MoleculeDataset


def cross_validate(args: Namespace, iteration: int, logger: Logger = None, dataset_loaded: MoleculeDataset = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Initialize relevant variables
    if args.change_seed:
        init_seed = args.seed + iteration
    else:
        init_seed = args.seed
    save_dir = args.save_dir
    if args.task_names == None:
        args.task_names = get_task_names(path=args.data_path, use_compound_names= args.use_compound_names, use_conf_id= args.use_conf_id)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    if dataset_loaded == None:
        data = get_data(path=args.data_path, args=args, logger=logger)
    else:
        data = dataset_loaded

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        _, test_scores, _, _ = run_training(args= args, logger = logger, dataset_loaded=data)
        all_scores.append(test_scores)
    all_scores = np.array(all_scores)

    avg_dir = os.path.join(save_dir, f'avg_iteration{iteration}')
    makedirs(avg_dir)
    writer = SummaryWriter(log_dir=avg_dir)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(args.task_names, scores):
                info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')
    writer.add_scalar(f'Average_test_{args.metric}_iterations', mean_score,iteration)

    if args.show_individual_scores:
        for task_num, task_name in enumerate(args.task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score, data
