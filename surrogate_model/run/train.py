from argparse import Namespace
import logging
from typing import Callable, List, Union
# import os
# import psutil

import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange

from utilities.data import MoleculeDataset
from utilities.utils_nn import compute_gnorm, compute_pnorm, NoamLR, SinexpLR, LRRange


def train(model: nn.Module,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          metric_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboard SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    
    model.train()
    
    data.shuffle()

    if args.single_mol_tasks:
        m_targets = [item for sublist in args.mol_targets for item in sublist]
    else:
        m_targets = args.mol_targets

    loss_sum, metric_sum, iter_count = [0]*(len(args.atom_targets) + len(args.bond_targets) + len(m_targets)), \
                                       [0]*(len(args.atom_targets) + len(args.bond_targets) + len(m_targets)), 0

    loss_weights = args.loss_weights

    num_iters = len(data) // args.batch_size * args.batch_size  # don't use the last batch if it's small, for stability

    iter_size = args.batch_size

    for i in trange(0, num_iters, iter_size):
        # mem = psutil.Process(os.getpid()).memory_info()[0]/(2.**30)
        # writer.add_scalar("Memory",mem, i)
        # Prepare batch
        if i + args.batch_size > len(data):
            break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets(scaled_targets=args.target_scaling, ind_props=args.single_mol_tasks)
        batch = smiles_batch
        #import pdb; pdb.set_trace()
        targets = [torch.Tensor(np.concatenate(x)) for x in zip(*target_batch)]
        if next(model.parameters()).is_cuda:
        #   mask, targets = mask.cuda(), targets.cuda()
            targets = [x.cuda() for x in targets]

        # Run model
        model.zero_grad(set_to_none=True)
        preds = model(batch, features_batch)
        preds_dim = [len(x[0]) for x in preds]
        targets = [x.reshape([-1, dim]) for x, dim in zip(targets, preds_dim)]

        loss_multi_task = []
        metric_multi_task = []
        for target, pred, lw in zip(targets, preds, loss_weights):
            loss = loss_func(pred, target)
            loss = loss.sum(dim=-2) / target.shape[0]
            lw = torch.Tensor([lw])
            if args.cuda:
                lw = lw.cuda()
            weighted_loss = loss * lw
            loss_multi_task.append((weighted_loss.squeeze()).mean())
            if args.cuda:
                metric = metric_func(pred.data.cpu().numpy(), target.data.cpu().numpy())
            else:
                metric = metric_func(pred.data.numpy(), target.data.numpy())
            metric_multi_task.append(metric)

        loss_sum = [x + y for x,y in zip(loss_sum, loss_multi_task)]
        iter_count += 1

        sum(loss_multi_task).backward()
        optimizer.step()

        metric_sum = [x + y for x,y in zip(metric_sum, metric_multi_task)]

        if isinstance(scheduler, NoamLR) or isinstance(scheduler, SinexpLR) or isinstance(scheduler, LRRange):
            scheduler.step()

        n_iter += args.batch_size

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = [x / iter_count for x in loss_sum]
            metric_avg = [x / iter_count for x in metric_sum]
            loss_sum, iter_count, metric_sum = [0]*(len(args.atom_targets) + len(args.bond_targets) + len(m_targets)), \
                                               0, \
                                               [0]*(len(args.atom_targets) + len(args.bond_targets) + len(m_targets))

            loss_str = ', '.join(f'lss_{i} = {lss:.4e}' for i, lss in enumerate(loss_avg))
            metric_str = ', '.join(f'mc_{i} = {mc:.4e}' for i, mc in enumerate(metric_avg))
            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'{loss_str}, {metric_str}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                for i, lss in enumerate(loss_avg):
                    writer.add_scalar(f'train_loss_{i}', lss, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
