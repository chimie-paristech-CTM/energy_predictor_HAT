import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data.dataset import Dataset


class PredDataset(Dataset):
    """A dataset object for the FFNN."""

    def __init__(
        self,
        df,
        features,
        target_column1='DG_TS_tunn',
        rxn_id_column='rxn_id',
        train=True,
    ):
        """
        Args:
            df (pd.DataFrame): dataframe containing rxn_id, rxn_smiles, targets and descriptors
            target_column1 (str): name of the first target column
            target_column2 (str): name of the second target column

        """
        self.train = train
        if train:
            self.target = torch.tensor(df[f"{target_column1}"].values, dtype=torch.float32)
        else:
            self.target = None
        self.descriptors = torch.tensor(df[features].values, dtype=torch.float32)

    def __len__(self):
        """__len__.
        """
        return self.descriptors.shape[0]

    def __getitem__(self, idx: int):
        """__getitem__.

        Args:
            idx (int): idx
        """
        x = self.descriptors[idx]
        if self.train:
            y = self.target[idx]
            return x, y
        else:
            return x

    def getdimension(self):
        """___getdimension
        """
        return self.descriptors.shape[-1]


def normalize_data(df, rxn_id_column, scalers=None):

    df_scaled = pd.DataFrame()

    df_scaled[f"{rxn_id_column}"] = df[f"{rxn_id_column}"]

    if scalers is None:
        scalers = {}
        for column in df.columns:
            if column != f"{rxn_id_column}":
                scaler = StandardScaler()
                data = df[column].values.reshape(-1, 1).tolist()

                scaler.fit(data)
                scalers[column] = scaler

    for column in df.columns:
        if column != f"{rxn_id_column}":
            scaler = scalers[column]
            df_scaled[column] = df[column].apply(lambda x: scaler.transform([[x]])[0])
            df_scaled[column] = df_scaled[column].apply(lambda x: x[0])

    return df_scaled, scalers


def scaling_back(test_pred, scaler):

    a = pd.DataFrame()

    for i in test_pred:
        if torch.cuda.is_available():
            i = i.cpu()
        a = pd.concat([a, pd.DataFrame(i.numpy())], axis=0, ignore_index=True)
    a.rename(columns={0: 'DG_TS_PRED'}, inplace=True)

    pred_value = scaler.inverse_transform(a['DG_TS_PRED'].array.reshape(-1, 1))
    return pred_value


def get_metrics(true, pred):

    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)

    return mae, rmse, r2


def split_data_cross_val(
    df,
    k_fold_arange,
    i,
    j,
    rxn_id_column,
    data_path,
    csv_file,
    train_valid_set_path,
    sample,
    k_fold,
    random_state,
    test_set_path,
):
    """Splits a dataframe into train, valid, test dataframes for a single fold, single model in cv

    Args:
        df (pd.DataFrame): the entire dataset
        k_fold_arange (np.linspace): limits of the individual folds
        i (int): current fold (cross-validation)
        j (int): current model (ensemble)
        rxn_id_column (str): the name of the rxn-id column in the dataframe
        data_path (str): path to the entire dataset
        train_valid_set_path (str): path the to train/valid dataset (selective sampling)
        sample (int): number of training points to sample
        k_fold (int): number of folds
        random_state (int): the random state to be used for the splitting and sampling
    """
    if data_path or csv_file is not None:
        test = df[k_fold_arange[i]: k_fold_arange[i + 1]]
        valid = df[~df[f"{rxn_id_column}"].isin(test[f"{rxn_id_column}"])].sample(
            frac=1 / (k_fold - 1), random_state=random_state + j
        )
        train = df[
            ~(
                df[f"{rxn_id_column}"].isin(test[f"{rxn_id_column}"])
                | df[f"{rxn_id_column}"].isin(valid[f"{rxn_id_column}"])
            )
        ]
    elif train_valid_set_path is not None and test_set_path is not None:
        valid = df.sample(frac=1 / (k_fold - 1), random_state=random_state + i + j)
        train = df[~(df[f"{rxn_id_column}"].isin(valid[f"{rxn_id_column}"]))]
        test = pd.read_csv(test_set_path, index_col=0)

    # down sample training and validation sets in case args.sample keyword has been selected
    if sample:
        try:
            train = train.sample(n=sample, random_state=random_state + j)
            valid = valid.sample(
                n=math.ceil(int(sample) / 4), random_state=random_state + j
            )
        except Exception:
            pass

    return train, valid, test


def split_data_training(df, rxn_id_column, splits, random_state, i):
    """Splits a dataframe into train, valid, test dataframes for a single model (regular training)

    Args:
        df (pd.DataFrame): the entire dataset
        rxn_id_column (str): the name of the rxn-id column in the dataframe
        splits (List[int]): relative sizes of train, valid, test sets
        random_state (int): random state used for data splitting
        i (int): current model (ensemble)
    """

    test_ratio = splits[2]/sum(splits)
    valid_ratio = splits[1]/sum(splits[:2])
    test = df.sample(frac=test_ratio, random_state=random_state)
    valid = df[~df[f"{rxn_id_column}"].isin(test[f"{rxn_id_column}"])].sample(
        frac=valid_ratio, random_state=random_state + i)
    train = df[
        ~(
            df[f"{rxn_id_column}"].isin(test[f"{rxn_id_column}"])
            | df[f"{rxn_id_column}"].isin(valid[f"{rxn_id_column}"])
        )]

    return train, valid, test


def write_predictions(
        rxn_ids,
        predicted_activation_energies,
        rxn_id_column,
        file_name,
):
    """Write predictions to a .csv file.

        Args:
            rxn_id (pd.DataFrame): dataframe consisting of the rxn_ids
            activation_energies_predicted (List): list of predicted activation energies
            reaction_energies_predicted (List): list of predicted reaction energies
            rxn_id_column (str): name of the rxn-id column
            file_name : name of .csv file to write the predicted values to
        """

    test_predicted = pd.DataFrame()
    test_predicted[f"{rxn_id_column}"] = rxn_ids
    test_predicted["predicted_activation_energy"] = predicted_activation_energies
    test_predicted.to_csv(file_name)

    return None
