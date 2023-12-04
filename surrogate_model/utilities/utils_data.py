from argparse import Namespace
import csv
from logging import Logger
import pickle
import random
from typing import List, Set, Tuple
import os

from rdkit import Chem
import numpy as np
from tqdm import tqdm
import pandas as pd

from .data import MoleculeDatapoint, MoleculeDataset
from .add_features import load_features


def get_task_names(path: str, args: Namespace) -> List[str]:
    """
    Gets the task names from a data pickle file.

    :param path: Path to a pickle file.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :param use_conf_id: Whether file has conformation IDs.
    :return: A list of task names.
    """
    header = get_header(path)

    task_names = []
    for task in args.atom_targets + args.bond_targets:
        if task in header:
            task_names.append(task)
    
    if args.single_mol_tasks:
        for task in args.mol_ext_targets + args.mol_int_targets:
            if task in header:
                task_names.append(task)
    else:
        if args.mol_ext_targets:
            task_names.append('EXT_MOL_PROPS')
        if args.mol_int_targets:
            task_names.append('INT_MOL_PROPS')
        
    return task_names


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data pickle file.

    :param path: Path to a pickle file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    data = pd.read_pickle(path)
    header = data.columns.to_list()

    return header


def get_smiles(path: str, header: bool = True) -> List[str]:
    """
    Returns the smiles strings from a data CSV file (assuming the first line is a header).

    :param path: Path to a CSV file.
    :param header: Whether the CSV file contains a header (that will be skipped).
    :return: A list of smiles strings.
    """
    with open(path) as f:
        reader = csv.reader(f)
        if header:
            next(reader)  # Skip header
        smiles = [line[0] for line in reader]

    return smiles


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A MoleculeDataset.
    :return: A MoleculeDataset with only valid molecules.
    """
    return MoleculeDataset([datapoint for datapoint in data
                            if datapoint.smiles != '' and datapoint.mol is not None
                            and datapoint.mol.GetNumHeavyAtoms() > 0]) 


def get_data(path: str,
             skip_invalid_smiles: bool = True,
             args: Namespace = None,
             features_path: List[str] = None,
             max_data_size: int = None,
             use_compound_names: bool = None,
             use_conf_id: bool = None,
             logger: Logger = None) -> MoleculeDataset:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a pickle file.

    :param path: Path to a pickle file.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param args: Arguments.
    :param features_path: A list of paths to files containing features. If provided, it is used
    in place of args.features_path.
    :param max_data_size: The maximum number of data points to load.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :param use_conf_id: Whether file has conf IDs in addition to smiles strings.
    :param logger: Logger.
    :return: A MoleculeDataset containing smiles strings and target values along
    with other info such as additional features and compound names when desired.
    """
    debug = logger.debug if logger is not None else print

    if args is not None:
        # Prefer explicit function arguments but default to args if not provided
        features_path = features_path if features_path is not None else args.features_path
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        use_compound_names = use_compound_names if use_compound_names is not None else args.use_compound_names
        use_conf_id = use_conf_id if use_conf_id is not None else args.use_conf_id
    else:
        use_compound_names = False
        use_conf_id = False

    max_data_size = max_data_size or float('inf')

    # Load additional features (contributed externally)
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    skip_smiles = set()

    # Load data
    data = pd.read_pickle(path)
    data = data[~data.smiles.isin(skip_smiles)]

    if len(data) > max_data_size:
        if args.datasubset_sample:
            frac = max_data_size/len(data)
            data = data.sample(frac=frac)
        else:
            data = data.iloc[:max_data_size]

    data = MoleculeDataset([
        MoleculeDatapoint(
            line=line,
            args=args,
            features=features_data[i] if features_data is not None else None,
            use_compound_names=use_compound_names,
            use_conf_id = use_conf_id
        ) for i, line in tqdm(data.iterrows())
    ])

    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)

    return data


def get_data_from_smiles(smiles: List[str], skip_invalid_smiles: bool = True, logger: Logger = None, args: Namespace = None) -> MoleculeDataset:
    """
    Converts SMILES to a MoleculeDataset.

    :param smiles: A list of SMILES strings.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param logger: Logger.
    :return: A MoleculeDataset with all of the provided SMILES.
    """
    debug = logger.debug if logger is not None else print

    data = MoleculeDataset([MoleculeDatapoint(line=pd.Series([smile], index=['smiles']), args=args, pred=True)
                            for smile in smiles])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               args: Namespace = None,
               logger: Logger = None) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
    """
    Splits data into training, validation, and test splits.

    :param data: A MoleculeDataset.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param seed: The random seed to use before shuffling data.
    :param args: Namespace of arguments.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    if args is not None:
        folds_file, val_fold_index, test_fold_index = \
            args.folds_file, args.val_fold_index, args.test_fold_index
    else:
        folds_file = val_fold_index = test_fold_index = None
    
    if split_type == 'crossval':
        index_set = args.crossval_index_sets[args.seed]
        data_split = []
        for split in range(3):
            split_indices = []
            for index in index_set[split]:
                with open(os.path.join(args.crossval_index_dir, f'{index}.pkl'), 'rb') as rf:
                    split_indices.extend(pickle.load(rf))
            data_split.append([data[i] for i in split_indices])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)
    
    elif split_type == 'index_predetermined':
        split_indices = args.crossval_index_sets[args.seed]
        assert len(split_indices) == 3
        data_split = []
        for split in range(3):
            data_split.append([data[i] for i in split_indices[split]])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'predetermined':
        if not val_fold_index:
            assert sizes[2] == 0  # test set is created separately so use all of the other data for train and val
        assert folds_file is not None
        assert test_fold_index is not None

        try:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f)
        except UnicodeDecodeError:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f, encoding='latin1')  # in case we're loading indices from python2
        # assert len(data) == sum([len(fold_indices) for fold_indices in all_fold_indices])

        folds = [[data[i] for i in fold_indices] for fold_indices in all_fold_indices]

        test = folds[test_fold_index]
        if val_fold_index is not None:
            val = folds[val_fold_index]

        train_val = []
        for i in range(len(folds)):
            if i != test_fold_index and (val_fold_index is None or i != val_fold_index):
                train_val.extend(folds[i])

        if val_fold_index is not None:
            train = train_val
        else:
            random.seed(seed)
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'random':
        data.shuffle(seed=seed)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'train_eq_test':
        data.shuffle(seed=seed)

        val_size = int(0.2 * len(data))

        train = data[:]
        val = data[:val_size]
        test = data[:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    else:
        raise ValueError(f'split_type "{split_type}" not supported.')

def flatten(t):
    '''
    Reduce dimension of a list.

    return: np array of reshaped list
    '''
    flat_list = [item for sublist in t for item in sublist]
    flat_array = [np.array(flat_list)]
    return flat_array