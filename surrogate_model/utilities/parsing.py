from argparse import ArgumentParser, Namespace
import json
import os
from tempfile import TemporaryDirectory
import pickle
from typing import List

import torch

from .utils_gen import makedirs

def add_predict_args(parser: ArgumentParser):
    """
    Adds predict arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--test_path', type=str,
                        help='Path to CSV file containing testing data for which predictions will be made')
    parser.add_argument('--path_bond_dict', type=str, default='all_bond_dict.pickle',
                        help='Path to pickle file with dictionary of the bond indices based on the atom map numbers.')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains CHEMBL ids in addition to SMILES strings and/or conformation id')
    parser.add_argument('--use_conf_id', action='store_true', default=False,
                        help='Use when test data file contains conformation ids in addition to SMILES strings and/or CHEMBL id')                    
    parser.add_argument('--preds_path', type=str,
                        help='Path to pickle file where predictions will be saved')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN')
    parser.add_argument('--no_features_scaling', action='store_true', default=False, # will only scale additional features
                        help='Turn off scaling of additional features')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    parser.add_argument('--datasubset_sample', action='store_true', default=False,
                        help='Shuffle data points if only use a subset (as in max_data_size).')
                        

def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--data_path', type=str,
                        help='Path to data pickle file')
    parser.add_argument('--path_bond_dict', type=str, default='all_bond_dict.pickle',
                        help='Path to pickle file with dictionary of the bond indices based on the atom map numbers.')
    parser.add_argument('--output', type=str,
                        help='prefix for output')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains CHEMBL ids in addition to SMILES strings and/or conformation id')
    parser.add_argument('--use_conf_id', action='store_true', default=False,
                        help='Use when test data file contains conformation ids in addition to SMILES strings and/or CHEMBL id')   
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    parser.add_argument('--datasubset_sample', action='store_true', default=False,
                        help='Shuffle data points if only use a subset (as in max_data_size).')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Whether to skip training and only test the model')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN')                   
    parser.add_argument('--save_dir', type=str, default='log',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--separate_val_path', type=str,
                        help='Path to separate val set, optional')
    parser.add_argument('--separate_val_features_path', type=str, nargs='*',
                        help='Path to file with features for separate val set')
    parser.add_argument('--separate_test_path', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--separate_test_features_path', type=str, nargs='*',
                        help='Path to file with features for separate test set')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'predetermined', 'crossval', 'index_predetermined', 'train_eq_test'],
                        help='Method of splitting the data into train/val/test.')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Number of folds when performing cross validation')
    parser.add_argument('--folds_file', type=str, default=None,
                        help='Optional file of fold labels')
    parser.add_argument('--val_fold_index', type=int, default=None,
                        help='Which fold to use as val for leave-one-out cross val')
    parser.add_argument('--test_fold_index', type=int, default=None,
                        help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--crossval_index_dir', type=str, 
                        help='Directory in which to find cross validation index files')
    parser.add_argument('--crossval_index_file', type=str, 
                        help='Indices of files to use as train/val/test'
                             'Overrides --num_folds and --seed.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.')
    parser.add_argument('--metric', type=str, default='mae',
                        choices=['rmse', 'mae', 'mse'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             'Note: Defaults to "mae" (regression).')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='The number of batches between each logging of the training loss')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual targets, not just average, at the end')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--config_path', type=str,
                        help='Path to a .json file containing arguments. Any arguments present in the config'
                             'file will override arguments specified via the command line or by the defaults.')
    parser.add_argument('--atom_targets', type=str, nargs='+', default=["GFN2:COVALENT_COORDINATION_NUMBER",
                                                                        "GFN2:DISPERSION_COEFFICIENT_ATOMIC",
                                                                        "GFN2:POLARIZABILITY_ATOMIC",
                                                                        "DFT:ESP_AT_NUCLEI",
                                                                        "DFT:LOWDIN_CHARGES",
                                                                        "DFT:MULLIKEN_CHARGES",
                                                                        "DFT:TOTAL_MAYER_BOND_ORDER",
                                                                        "DFT:TOTAL_WIBERG_LOWDIN_BOND_ORDER",
                                                                        "SASA",
                                                                        "PINT"],
                        help='training atom targets')
    parser.add_argument('--atom_constraints', type=float, nargs='*', default=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        help='Constraints applied to model output (0 for no constraint, 1 for charge constraint.')
    parser.add_argument('--bond_targets', type=str, nargs='*', default=["DFT:MAYER_BOND_ORDER",
                                                                        "DFT:WIBERG_LOWDIN_BOND_ORDER"],
                        help='training bond targets')
    parser.add_argument('--bond_constraints', type=float, nargs='*',
                        help='training bond constraints')
    parser.add_argument('--mol_ext_targets', type=str, nargs='*', default=["GFN2:TOTAL_ENTHALPY",
                                                                            "GFN2:TOTAL_FREE_ENERGY",
                                                                            "GFN2:ENTROPY",
                                                                            "DFT:FORMATION_ENERGY"
                                                                            ],
                        help='training extensive molecule targets')
    parser.add_argument('--mol_int_targets', type=str, nargs='*', default=["DFT:HOMO_ENERGY",
                                                                            "DFT:LUMO_ENERGY",
                                                                            "DFT:HOMO_LUMO_GAP"],
                        help='training intensive molecule targets')                   
    parser.add_argument('--loss_weights', type=float, nargs='*',
                        help='Weights for multi-task loss. Must be provided if "no_equal_lss_weights" is set to True.')
    parser.add_argument('--no_equal_lss_weights', action='store_true', default=False,
                        help='Turn on equal loss weights for all target properties.')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--lr_schedule', type=str, default='Sinexp', choices=['Sinexp','Noam','LRRange'],
                        help= 'Type of learning rate schedule.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate (also for Sinexp)')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate (only used in Noam)')
    parser.add_argument('--final_lr', type=float, default=1e-5,
                        help='Final learning rate')
    parser.add_argument('--single_mol_tasks', action='store_true', default=False,
                        help='Do single task learning also for molecular properties (individual FFNs).')
    parser.add_argument('--no_features_scaling', action='store_true', default=False, # will only scale addtional features
                        help='Turn off scaling of additional features')
    parser.add_argument('--target_scaling', action='store_true', default=False, # will only scale addtional features
                        help='Turn on scaling of training targets.')
    parser.add_argument('--target_scaler_type', default=None,
                        choices=['MinMaxScaler', 'StandardScaler'],
                        help='Choose type of scaling of the targets.')
    parser.add_argument('--atom_wise_scaling', type = int, nargs='*', default = [0,1,1,1,0,0,0,0,0,0],
                        help= 'Atom properties whose target should be scaled per atom (order as in atom_targets). 1: atom-wise, 0: global') 
    parser.add_argument('--no_atom_scaling', action='store_true', default=False,
                        help='Turn off atom scaling')                                  
    parser.add_argument('--early_stopping', action='store_true', default=False,
                        help='Turn on early stopping with patience defined separately (default: 5).')
    parser.add_argument('--patience', type=int, default=5,
                        help='Set patience for early stopping.')


    # Model arguments
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')                     
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='Use messages on atoms instead of messages on bonds')
    parser.add_argument('--explicit_Hs', action='store_true', default=False,
                        help='Use explicit H atoms in the model')


def add_hyperopt_args(parser: ArgumentParser):
    """
    Adds hyperparameter optimization arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """  
    parser.add_argument('--num_iters', type=int, default=20,
                        help='Number of hyperparameter choices to try.') 
    parser.add_argument('--config_save_path', type=str,
                        help='Path to :code:`.json` file where best hyperparameter settings will be written.')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='(Optional) Path to a directory where all results of the hyperparameter optimization will be written.')
    parser.add_argument('--hyperopt_checkpoint_dir', type=str, default=None,
                        help='Path to a directory where hyperopt completed trial data is stored. Hyperopt job will include these trials if restarted. \
                            Can also be used to run multiple instances in parallel if they share the same checkpoint directory.')
    parser.add_argument('--startup_random_iters', type=int, default=10,
                        help='The initial number of trials that will be randomly specified before TPE algorithm is used to select the rest.')    
    parser.add_argument('--manual_trial_dirs', type=List[str], default=None,
                        help='Paths to save directories for manually trained models in the same search space as the hyperparameter search.\
                            Results will be considered as part of the trial history of the hyperparameter search.') 
    parser.add_argument('--change_seed', action='store_true', default=False,
                        help='Whether to change seed every iteration of hyperparameter change. Do not use if num_folds > 1.')


def update_checkpoint_args(args: Namespace):
    """
    Walks the checkpoint directory to find all checkpoints, updating args.checkpoint_paths and args.ensemble_size.

    :param args: Arguments.
    """
    if hasattr(args, 'checkpoint_paths') and args.checkpoint_paths is not None:
        return

    if args.checkpoint_dir is not None and args.checkpoint_path is not None:
        raise ValueError('Only one of checkpoint_dir and checkpoint_path can be specified.')

    if args.checkpoint_dir is None:
        args.checkpoint_paths = [args.checkpoint_path] if args.checkpoint_path is not None else None
        return

    args.checkpoint_paths = []

    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))

    args.ensemble_size = len(args.checkpoint_paths)

    if args.ensemble_size == 0:
        raise ValueError(f'Failed to find any model checkpoints in directory "{args.checkpoint_dir}"')


def modify_predict_args(args: Namespace):
    """
    Modifies and validates predicting args in place.

    :param args: Arguments.
    """
    assert args.test_path
    assert args.preds_path
    assert args.checkpoint_dir is not None or args.checkpoint_path is not None or args.checkpoint_paths is not None

    update_checkpoint_args(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    args.predict_only = True
    # Create directory for preds path
    makedirs(args.preds_path, isfile=True)


def parse_predict_args() -> Namespace:
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args()
    modify_predict_args(args)

    return args


def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """
    global temp_dir  # Prevents the temporary directory from being deleted upon function return

    # Load config file
    if args.config_path is not None:
        with open(args.config_path) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args, key, value)
    
    assert args.data_path is not None

    if args.save_dir is not None:
        makedirs(args.save_dir)
    else:
        temp_dir = TemporaryDirectory()
        args.save_dir = temp_dir.name

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    if 'none' in args.atom_targets:
        args.atom_targets = []
    if 'none' in args.bond_targets:
        args.bond_targets = []
    if 'none' in args.mol_ext_targets:
        args.mol_ext_targets = []
    if 'none' in args.mol_int_targets:
        args.mol_int_targets = []

    if args.atom_constraints is not None:
        args.atom_constraints = torch.Tensor(args.atom_constraints)
        if args.cuda:
            args.atom_constraints = args.atom_constraints.cuda()

    args.mol_targets = [props for props in [args.mol_ext_targets, args.mol_int_targets] if props]
    
    args.equal_lss_weights = not args.no_equal_lss_weights

    len_a = len(args.atom_targets)
    len_b = len(args.bond_targets)
    len_ab = len_a+len_b
    len_mol_ext = len(args.mol_ext_targets)
    len_mol_int = len(args.mol_int_targets)

    if args.loss_weights is not None:
        if not args.single_mol_tasks:
            args.loss_weights = args.loss_weights[:len_ab]+ [args.loss_weights[len_ab:len_ab+len_mol_ext]]+ [args.loss_weights[len_ab+len_mol_ext:]]
        args.loss_weights = [x for x in args.loss_weights if x]
    elif args.equal_lss_weights:
        if not args.single_mol_tasks:
            args.loss_weights = [1] * len_ab + [[1] * len_mol_ext] + [[1] * len_mol_int]
            args.loss_weights = [x for x in args.loss_weights if x]
        else:
            args.loss_weights = [1] * (len_ab + len_mol_ext + len_mol_int)

    if args.bond_constraints is not None:
        args.bond_constraints = torch.Tensor(args.bond_constraints)
        if args.cuda:
            args.bond_constraints = args.bond_constraints.cuda()

    args.features_scaling = not args.no_features_scaling
    del args.no_features_scaling

    update_checkpoint_args(args)

    args.use_input_features = args.features_path

    args.num_lrs = 1

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.hidden_size

    assert (args.split_type == 'predetermined') == (args.folds_file is not None) == (args.test_fold_index is not None)
    assert (args.split_type == 'crossval') == (args.crossval_index_dir is not None)
    assert (args.split_type in ['crossval', 'index_predetermined']) == (args.crossval_index_file is not None)
    if args.split_type in ['crossval', 'index_predetermined']:
        with open(args.crossval_index_file, 'rb') as rf:
            args.crossval_index_sets = pickle.load(rf)
        args.num_folds = len(args.crossval_index_sets)
        args.seed = 0

    if args.test:
        args.epochs = 0
    
    args.task_names = None

    if not args.early_stopping:
        del args.patience

    args.predict_only = False

    if args.no_atom_scaling:
        args.atom_wise_scaling = []
    elif not args.atom_targets:
        args.atom_wise_scaling = []
    else:
        assert len(args.atom_wise_scaling) == len(args.atom_targets)

def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    return args


def modify_hyperopt_args(args: Namespace):
    """
    Modifies and validates hyperparameter optimization arguments in place.

    :param args: Arguments.
    """    
    if args.log_dir is None:
        args.log_dir = args.save_dir
    if args.hyperopt_checkpoint_dir is None:
        args.hyperopt_checkpoint_dir = args.log_dir
    if args.num_folds > 1:
        assert args.change_seed == False


def parse_hyperopt_args() -> Namespace:
    """
    Parses arguments for hyperparameter optimization.

    :return: A Namespace containing the parsed, modified, and validated args.
    """    
    parser = ArgumentParser()
    add_hyperopt_args(parser)
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)
    modify_hyperopt_args(args)

    return args
