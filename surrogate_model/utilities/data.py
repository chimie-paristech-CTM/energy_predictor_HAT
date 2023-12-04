from argparse import Namespace
import random
from typing import Callable, List, Union

import numpy as np
from torch.utils.data.dataset import Dataset
from rdkit import Chem
import pandas as pd

from .utils_map import atom_map_dict
from .scaler import StandardScaler


class MoleculeDatapoint:
    """A MoleculeDatapoint contains a single molecule and its associated features and targets."""

    def __init__(self,
                 line: pd.Series,
                 args: Namespace = None,
                 features: np.ndarray = None,
                 use_compound_names: bool = False,
                 use_conf_id: bool = False,
                 pred: bool = False):
        """
        Initializes a MoleculeDatapoint, which contains a single molecule.

        :param line: a pandas Series
        :param args: Arguments.
        :param features: A numpy array containing additional features (ex. Morgan fingerprint).
        :param use_compound_names: Whether the data file includes the CHEMBL id on each line.
        :param use_conf_id: Whether the data file includes the conf id on each line.
        """
        if args is not None:
            self.args = args
        else:
            self.args = None

        self.features = features

        if use_compound_names:
            self.compound_name = line['CHEMBL_ID']  # str
        else:
            self.compound_name = None

        if use_conf_id:
            self.conf_id = line['CONF_ID']  # str
        else:
            self.conf_id = None

        self.smiles = line['smiles']  # str

        smiles_parser = Chem.SmilesParserParams()
        smiles_parser.removeHs = not args.explicit_Hs

        self.mol = Chem.MolFromSmiles(self.smiles, smiles_parser)

        if not args.no_atom_scaling:
            _, mol = atom_map_dict(self.mol, includeHs=args.explicit_Hs)
            self.atom_types = mol.GetNumAtoms() * [np.nan]
            for a in mol.GetAtoms():
                self.atom_types[a.GetAtomMapNum()-1] = a.GetAtomicNum()
        else:
            self.atom_types = []

        # Fix nans in features
        if self.features is not None:
            replace_token = 0
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        if pred:
            return

        # Create atom, bond and molecular targets
        from .utils_data import flatten
        self.atom_targets = line[args.atom_targets].values.tolist()
        self.atom_targets = [np.array(a) for a in self.atom_targets]
        self.bond_targets = line[args.bond_targets].values.tolist()
        self.bond_targets = [np.array(a) for a in self.bond_targets]
        self.mol_targets = line[args.mol_ext_targets + args.mol_int_targets].values.tolist()
        if args.mol_ext_targets:
            self.mol_ext_targets = flatten(line[args.mol_ext_targets].values.tolist())
        else:
            self.mol_ext_targets = []
        if args.mol_int_targets:
            self.mol_int_targets = flatten(line[args.mol_int_targets].values.tolist())
        else:
            self.mol_int_targets = []

    def set_features(self, features: np.ndarray):
        """
        Sets the features of the molecule.

        :param features: A 1-D numpy array of features for the molecule.
        """
        self.features = features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        if self.args.single_mol_tasks:
            mol_tasks = len(self.mol_targets)
        else:
            if self.mol_ext_targets and self.mol_int_targets:
                mol_tasks = 2
            elif self.mol_ext_targets or self.mol_int_targets:
                mol_tasks = 1
            else:
                mol_tasks = 0

        return len(self.atom_targets) + len(self.bond_targets) + mol_tasks

    def set_targets(self, targets: List[float]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.scaled_targets = targets


class MoleculeDataset(Dataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, data: List[MoleculeDatapoint]):
        """
        Initializes a MoleculeDataset, which contains a list of MoleculeDatapoints (i.e. a list of molecules).

        :param data: A list of MoleculeDatapoints.
        """
        self.data = data
        self.args = self.data[0].args if len(self.data) > 0 else None
        self.scaler = None

    def compound_names(self) -> List[str]:
        """
        Returns the CHEMBL ids associated with the molecule (if they exist).

        :return: A list of compound names or None if the dataset does not contain compound names.
        """
        if len(self.data) == 0 or self.data[0].compound_name is None:
            return None

        return [d.compound_name for d in self.data]

    def conf_ids(self) -> List[str]:
        """
        Returns the conf ids associated with the molecule (if they exist).

        :return: A list of compound names or None if the dataset does not contain compound names.
        """
        if len(self.data) == 0 or self.data[0].conf_id is None:
            return None

        return [d.conf_id for d in self.data]

    def smiles(self) -> List[str]:
        """
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        """
        return [d.smiles for d in self.data]
    
    def mols(self) -> List[Chem.Mol]:
        """
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        """
        return [d.mol for d in self.data]

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        return [d.features for d in self.data]

    def targets(self, scaled_targets = False, ind_props = False) -> List[List[float]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats containing the targets.
        """
        if scaled_targets:
            targets_list = [d.scaled_targets for d in self.data]
        elif ind_props:
            targets_list = [d.atom_targets + d.bond_targets + d.mol_targets 
                for d in self.data]
        else:
            targets_list = [d.atom_targets + d.bond_targets + d.mol_ext_targets + d.mol_int_targets
                for d in self.data]

        return targets_list

    def get_atom_types(self) -> List[List[int]]:
        '''
        Returns the atom types (as atomic numbers) associated with each molecule.

        :return: A list of lists of integers describing the atomic number.
        '''
        return [d.atom_types for d in self.data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self.data[0].num_tasks() if len(self.data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the features array associated with each molecule.

        :return: The size of the features.
        """
        return len(self.data[0].features) if len(self.data) > 0 and self.data[0].features is not None else None

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
    

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
        Normalizes the features of the dataset using a StandardScaler (subtract mean, divide by standard deviation).

        If a scaler is provided, uses that scaler to perform the normalization. Otherwise fits a scaler to the
        features in the dataset and then performs the normalization.

        :param scaler: A fitted StandardScaler. Used if provided. Otherwise a StandardScaler is fit on
        this dataset and is then used.
        :param replace_nan_token: What to replace nans with.
        :return: A fitted StandardScaler. If a scaler is provided, this is the same scaler. Otherwise, this is
        a scaler fit on this dataset.
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None # this will be returned when no additional features (for scaling of features not targets)

        if scaler is not None:
            self.scaler = scaler

        elif self.scaler is None:
            features = np.vstack([d.features for d in self.data])
            self.scaler = StandardScaler(replace_nan_token=replace_nan_token, scale_features=True)
            self.scaler.fit(features)

        for d in self.data:
            d.set_features(self.scaler.transform(d.features.reshape(1, -1))[0])

        return self.scaler
    
    def set_targets(self, targets: List[List[float]]):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats containing targets for each molecule. This must be the
        same length as the underlying dataset.
        """
        assert len(self.data) == len(targets)
        for i in range(len(self.data)):
            self.data[i].set_targets(targets[i])

    def sort(self, key: Callable):
        """
        Sorts the dataset using the provided key.

        :param key: A function on a MoleculeDatapoint to determine the sorting order.
        """
        self.data.sort(key=key)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e. the number of molecules).

        :return: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        """
        Gets one or more MoleculeDatapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A MoleculeDatapoint if an int is provided or a list of MoleculeDatapoints if a slice is provided.
        """
        return self.data[item]
