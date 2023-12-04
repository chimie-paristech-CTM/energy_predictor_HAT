"""Trains a model on a dataset."""

from utilities.parsing import parse_train_args
from run.run_training import run_training
from utilities.utils_gen import create_logger
import pickle

import os

if __name__ == '__main__':
    args = parse_train_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    avg_test_score, test_scores, test_preds, test_smiles = run_training(args, logger)
    with open(os.path.join(args.save_dir, 'test_preds.pickle'), 'wb') as preds:
        pickle.dump(test_preds, preds)

    with open(os.path.join(args.save_dir, 'test_smiles.pickle'), 'wb') as smiles:
        pickle.dump(test_smiles, smiles)
