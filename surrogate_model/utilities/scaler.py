from typing import Any, List

import numpy as np

def _map_stats(x, dict_a, scaler):
    new_x = np.empty(np.shape(x))
    for i, elem in enumerate(x):
        try:
            new_x[i] = dict_a[elem]
        except: # element was not in training set
            if scaler == 'mean':
                new_x[i] = 0
            elif scaler == 'std':
                new_x[i] = 1
            elif scaler == 'minmax':
                new_x[i] = 0 # will result in nan, which gets correct to zero subseqeuntly
            else:
                raise ValueError('Unknown scaler')

    return new_x

class StandardScaler:
    """A StandardScaler normalizes a dataset.

    When fit on a dataset, the StandardScaler learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the StandardScaler subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, means_atom: List = None, 
                stds_atom: List = None, replace_nan_token: Any = None, scale_features = False, atom_wise=[]):
        """
        Initialize StandardScaler, optionally with means and standard deviations precomputed.

        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: The token to use in place of nans.
        :param scale_features: Default applies when scaling targets (different shape and heterogenous dimensions).
        """
        self.means = means
        self.stds = stds
        self.means_atom = means_atom
        self.stds_atom = stds_atom
        self.atom_wise = atom_wise
        self.scale_features = scale_features
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[float]], len_ext =None, len_int=None, atom_types:List[List[int]]=None, mol_multitasks=True) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis.

        :param X: A list of lists of floats.
        :param len_ext: Number (int) of extensive molecular properties to predict.
        :param len_int: Number (int) of intensive molecular properties to predict.
        :param atom_wise: List of integers describing, which atom properties to scale atom-wise (1=True, 0=False)
        :param atom_types: List of lists of integers with the atomic numbers of all the passed molecules.
        :return: The fitted StandardScaler.
        """         
        if self.scale_features:
            X = np.array(X).astype(float)
            self.means = np.nanmean(X, axis=0)
            self.stds = np.nanstd(X, axis=0)
            self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
            self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
            self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)
        else:
            X = np.array(X, dtype=object)
            means_list = []
            stds_list = []
            for c_idx in range(len(X[0])):
                col = X[:, c_idx]
                col_ext = np.concatenate(col)
                mean = np.nanmean(col_ext, axis=0, dtype=np.float64)
                std = np.nanstd(col_ext, axis=0, dtype=np.float64)
                means_list.append(mean)
                stds_list.append(std)
            self.means = np.array(means_list)
            self.stds = np.array(stds_list)
            self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
            self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
            self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

            self.means_atom = len(self.means) * [np.nan]
            self.stds_atom = len(self.stds) * [np.nan]
            # scaling based on atom type
            for i, pos in enumerate(self.atom_wise):
                if pos == 1:
                    atom_to_mean = {}
                    atom_to_std = {}
                    col = X[:, i]
                    values = np.concatenate(col)
                    keys = np.concatenate(np.array(atom_types, dtype='object'))
                    atom_to_value = {}
                    for (key, value) in zip(keys,values):
                        if key in atom_to_value:
                            atom_to_value[key].append(value)
                        else:
                            atom_to_value[key] = [value]
                    for key, value in atom_to_value.items():
                        mean = np.nanmean(np.array(value), axis=0)
                        std = np.nanstd(np.array(value), axis=0)
                        if std == 0:
                            std = 1.0
                        atom_to_mean[key] = mean
                        atom_to_std[key] = std
                    self.means_atom[i] = atom_to_mean
                    self.stds_atom[i] = atom_to_std

            if len_ext + len_int > 0 and mol_multitasks:
                a_b_means = self.means[:-(len_ext+len_int)].tolist()
                ext_means = np.reshape(self.means[-(len_ext+len_int):-len_int], (-1)).tolist()
                int_means = np.reshape(self.means[-len_int:], (-1)).tolist()
                a_b_means.append(ext_means)
                a_b_means.append(int_means)
                self.means = np.array([x for x in a_b_means if x], dtype=object)
                a_b_stds = self.stds[:-(len_ext+len_int)].tolist()
                ext_stds = np.reshape(self.stds[-(len_ext+len_int):-len_int], (-1)).tolist()
                int_stds = np.reshape(self.stds[-len_int:], (-1)).tolist()
                a_b_stds.append(ext_stds)
                a_b_stds.append(int_stds)
                self.stds = np.array([x for x in a_b_stds if x], dtype=object)

        return self

    def transform(self, X: List[List[float]], atom_types:List[List[int]]=None):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats.
        :return: The transformed data.
        """
        if self.scale_features:
            X = np.array(X).astype(float)
            transformed_with_nan = (X - self.means) / self.stds
            transformed = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
        else:
            X = np.array(X, dtype=object)
            try:
                transformed = (X - self.means.reshape(-1,1)) / self.stds.reshape(-1,1)
                if X.shape != transformed.shape:
                    transformed = (X - self.means) / self.stds
            except:
                transformed = (X - self.means) / self.stds
            if self.atom_wise:
                pad = X.shape[1] - len(self.atom_wise)
                atom_wise_padded = self.atom_wise + pad * [0]
                for i, prop in enumerate(atom_wise_padded):
                    if prop == 1:
                        mapped_means = np.array([_map_stats(a, self.means_atom[i], scaler='mean') for a in atom_types], dtype='object')
                        mapped_stds = np.array([_map_stats(a, self.stds_atom[i], scaler='std') for a in atom_types], dtype='object')
                        transformed[:,i] = np.divide((X[:,i]-mapped_means), mapped_stds)

        return transformed

    def inverse_transform(self, X: List[List[float]], atom_types:List[List[int]]=None):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data.
        """
        if self.scale_features:
            X = np.array(X).astype(float)
            transformed_with_nan = X * self.stds + self.means
            transformed = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
        else:
            try:
                X = np.array(X, dtype='object')
                if self.means.shape == X.shape:
                    transformed = (X * self.stds) + self.means
                else:
                    try:
                        transformed = (X * self.stds.reshape(-1,1)) + self.means.reshape(-1,1)
                    except:
                        transformed = (X * self.stds.reshape(X.shape[0],-1,1)) + self.means.reshape(X.shape[0],-1,1)
            except:
                transformed = []
                for col, std, mean in zip(X, self.stds, self.means):
                    col = (col * std) + mean
                    transformed.append(col)
                # transformed = (X * self.stds) + self.means
                
            if self.atom_wise:
                pad = X.shape[0] - len(self.atom_wise)
                atom_wise_padded = self.atom_wise + pad * [0]
                for i, prop in enumerate(atom_wise_padded):
                    if prop == 1:
                        mapped_means = np.array([_map_stats(a, self.means_atom[i], scaler='mean') for a in atom_types], dtype='object')
                        mapped_stds = np.array([_map_stats(a, self.stds_atom[i], scaler='std') for a in atom_types], dtype='object')
                        mapped_means = np.concatenate(mapped_means).reshape(X[i].shape)
                        mapped_stds = np.concatenate(mapped_stds).reshape(X[i].shape)
                        transformed[i] = (X[i] * mapped_stds) + mapped_means

        return transformed

class MinMaxScaler:
    """A MinMaxScaler normalizes a dataset to a range between 0 and 1.

    When fit on a dataset, the MinMaxScaler learns the minimum and maximum across the 0th axis.
    When transforming a dataset, the MinMaxScaler subtracts the min and divides by the difference of max - min.
    """

    def __init__(self, mins: np.ndarray = None, maxs: np.ndarray = None, mins_atom: List = None, 
                maxs_atom: List = None, replace_nan_token: Any = None, scale_features = False, atom_wise = []):
        """
        Initialize MinMaxScaler, optionally with means and standard deviations precomputed.

        :param min: An optional 1D numpy array of precomputed mins.
        :param max: An optional 1D numpy array of precomputed maxs.
        :param replace_nan_token: The token to use in place of nans.
        :param scale_features: Default applies when normalizing targets (different shape and heterogenous dimensions).
        """
        self.mins = mins
        self.maxs = maxs
        self.mins_atom = mins_atom
        self.maxs_atom = maxs_atom
        self.atom_wise = atom_wise
        self.scale_features = scale_features
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[float]], len_ext =None, len_int=None, atom_types:List[List[int]]=None, mol_multitasks = True) -> 'MinMaxScaler':
        """
        Learns min's and max's across the 0th axis.

        :param X: A list of lists of floats.
        :param len_ext: Number (int) of extensive molecular properties to predict.
        :param len_int: Number (int) of intensive molecular properties to predict.
        :param atom_wise: List of integers describing, which atom properties to scale atom-wise (1=True, 0=False)
        :param atom_types: List of lists of integers with the atomic numbers of all the passed molecules.
        :return: The fitted MinMaxScaler.
        """
        if self.scale_features:
            X = np.array(X).astype(float)
            self.mins = np.nanmin(X, axis=0)
            self.maxs = np.nanmax(X, axis=0)
            self.mins = np.where(np.isnan(self.mins), np.zeros(self.mins.shape), self.mins)
            self.maxs = np.where(np.isnan(self.maxs), np.ones(self.maxs.shape), self.maxs)
            self.maxs = np.where(self.maxs == 0, np.ones(self.maxs.shape), self.maxs)
        else:
            X = np.array(X, dtype=object)
            mins_list = []
            maxs_list = []
            for c_idx in range(len(X[0])):
                col = X[:, c_idx]
                col_ext = np.concatenate(col)
                min_col = np.nanmin(col_ext, axis=0)
                max_col = np.nanmax(col_ext, axis=0)
                mins_list.append(min_col)
                maxs_list.append(max_col)
            self.mins = np.array(mins_list)
            self.maxs = np.array(maxs_list)

            self.mins_atom = len(self.mins) * [np.nan]
            self.maxs_atom = len(self.maxs) * [np.nan]
            # scaling based on atom type
            for i, pos in enumerate(self.atom_wise):
                if pos == 1:
                    atom_to_mins = {}
                    atom_to_maxs = {}
                    col = X[:, i]
                    values = np.concatenate(col)
                    keys = np.concatenate(np.array(atom_types))
                    atom_to_value = {}
                    for (key, value) in zip(keys,values):
                        if key in atom_to_value:
                            atom_to_value[key].append(value)
                        else:
                            atom_to_value[key] = [value]
                    for key, value in atom_to_value.items():
                        minim = np.nanmin(np.array(value), axis=0)
                        maxim = np.nanmax(np.array(value), axis=0)
                        atom_to_mins[key] = minim
                        atom_to_maxs[key] = maxim
                    self.mins_atom[i] = atom_to_mins
                    self.maxs_atom[i] = atom_to_maxs

            if len_ext + len_ext > 0 and mol_multitasks:
                a_b_mins = self.mins[:-(len_ext+len_int)].tolist()
                ext_mins = np.reshape(self.mins[-(len_ext+len_int):-len_int], (-1))#.tolist()
                int_mins = np.reshape(self.mins[-len_int:], (-1))#.tolist()
                a_b_mins.append(ext_mins)
                a_b_mins.append(int_mins)
                self.mins = np.array([x for x in a_b_mins if x], dtype=object)
                a_b_maxs = self.maxs[:-(len_ext+len_int)].tolist()
                ext_maxs = np.reshape(self.maxs[-(len_ext+len_int):-len_int], (-1))#.tolist()
                int_maxs = np.reshape(self.maxs[-len_int:], (-1))#.tolist()
                a_b_maxs.append(ext_maxs)
                a_b_maxs.append(int_maxs)
                self.maxs = np.array([x for x in a_b_maxs if x], dtype=object)

        return self

    def transform(self, X: List[List[float]], atom_types:List[List[int]]=None):
        """
        Transforms the data by subtracting the minimums and dividing by the difference of the maximums - minimums.

        :param X: A list of lists of floats.
        :return: The transformed data.
        """
        if self.scale_features:
            X = np.array(X).astype(float)
            transformed_with_nan = (X - self.mins) / (self.maxs-self.mins)
            transformed = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
        else:
            X = np.array(X, dtype=object)
            transformed = (X - self.mins) / (self.maxs-self.mins)
            if self.atom_wise:
                pad = X.shape[1] - len(self.atom_wise)
                atom_wise_padded = self.atom_wise + pad * [0]
                for i, prop in enumerate(atom_wise_padded):
                    if prop == 1:
                        mapped_mins = np.array([_map_stats(a, self.mins_atom[i], scaler='minmax') for a in atom_types], dtype='object')
                        mapped_maxs = np.array([_map_stats(a, self.maxs_atom[i], scaler='minmax') for a in atom_types], dtype='object')
                        transformed[:,i] = np.divide((X[:,i]-mapped_mins), (mapped_maxs-mapped_mins))
                        transformed[:,i] = np.where(np.isnan(transformed[:,i]), 0, transformed[:,i])

        return transformed

    def inverse_transform(self, X: List[List[float]], atom_types:List[List[int]]=None):
        """
        Performs the inverse transformation by multiplying by the range of max to min and adding the minimums.

        :param X: A list of lists of floats.
        :return: The inverse transformed data.
        """
        if self.scale_features:
            X = np.array(X).astype(float)
            transformed_with_nan = X * (self.maxs-self.mins) + self.mins
            transformed = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
        else:
            X = np.array(X, dtype=object)
            transformed = X * (self.maxs-self.mins) + self.mins
            if self.atom_wise:
                pad = X.shape[0] - len(self.atom_wise)
                atom_wise_padded = self.atom_wise + pad * [0]
                for i, prop in enumerate(atom_wise_padded):
                    if prop == 1:
                        mapped_mins = np.array([_map_stats(a, self.mins_atom[i], scaler='minmax') for a in atom_types], dtype='object')
                        mapped_maxs = np.array([_map_stats(a, self.maxs_atom[i], scaler='minmax') for a in atom_types], dtype='object')
                        mapped_mins = np.concatenate(mapped_mins).reshape(X[i].shape)
                        mapped_maxs = np.concatenate(mapped_maxs).reshape(X[i].shape)
                        transformed[i] = np.multiply(X[i], (mapped_maxs-mapped_mins)) + mapped_mins

        return transformed

