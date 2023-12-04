from argparse import Namespace
import csv
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem

from run.predict import predict
from utilities.data import MoleculeDataset
from utilities.utils_data import get_data, get_data_from_smiles
from utilities.utils_gen import load_args, load_checkpoint, load_scalers


def make_predictions(args: Namespace, smiles: List[str] = None) -> List[Optional[List[float]]]:
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: A list of lists of target predictions.
    """
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])

    if not hasattr(train_args, 'single_mol_tasks'): 
            train_args.single_mol_tasks = False

    # Update args with training arguments
    for key, value in vars(train_args).items(): 
        if not hasattr(args, key): 
            setattr(args, key, value) 

    print('Loading data')
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False, args=args) # converts smiles to a MoleculeDataset
    else:
        test_data = get_data(path=args.test_path, args=args, use_compound_names=args.use_compound_names, use_conf_id= args.use_conf_id, skip_invalid_smiles=False)

    print('Validating SMILES')
    valid_indices = [i for i in range(len(test_data)) if Chem.MolFromSmiles(test_data[i].smiles) is not None]
    full_data = test_data
    test_data = MoleculeDataset([test_data[i] for i in valid_indices])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if train_args.features_scaling:
        test_data.normalize_features(features_scaler)

    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    #for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        # Load model
    model = load_checkpoint(args.checkpoint_path, cuda=args.cuda, predict_only = args.predict_only)
    test_preds, test_smiles_batch = predict(
        model=model,
        data=test_data,
        batch_size=args.batch_size,
        scaler=scaler
    )

    return test_preds, test_smiles_batch