import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def cross_val(df, model, n_folds, target_column='DG_TS_tunn', sample=None, split_dir=None):
    """
    Function to perform cross-validation

    Args:
        df (pd.DataFrame): the DataFrame containing features and targets
        model (sklearn.Regressor): An initialized sklearn model
        n_folds (int): the number of folds
        target_column (str): target column
        sample(int): the size of the subsample for the training set (default = None)
        split_dir (str): the path to a directory containing data splits. If None, random splitting is performed.

    Returns:
        int: the obtained RMSE and MAE
    """
    rmse_list, mae_list, r2_list = [], [], []
    feature_names = [column for column in df.columns if column not in['rxn_id', 'DG_TS', 'G_r', 'DG_TS_tunn', 'Barrier', 'file_name']]

    if split_dir == None:
        df = df.sample(frac=1, random_state=0)
        chunk_list = np.array_split(df, n_folds)

    for i in range(n_folds):
        if split_dir == None:
            df_train = pd.concat([chunk_list[j] for j in range(n_folds) if j != i])
            if sample != None:
                df_train = df_train.sample(n=sample)
            df_test = chunk_list[i]
        else:
            rxn_ids_train1 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/train.csv'))[['rxn_id']].values.tolist()
            rxn_ids_train2 = pd.read_csv(os.path.join(split_dir, f'fold_{i}/valid.csv'))[['rxn_id']].values.tolist()
            rxn_ids_train = list(np.array(rxn_ids_train1 + rxn_ids_train2).reshape(-1))
            df['train'] = df['rxn_id'].apply(lambda x: int(x) in rxn_ids_train)
            df_train = df[df['train'] == True]
            df_test = df[df['train'] == False]

        X_train, y_train = df_train[feature_names], df_train[[target_column]]
        X_test, y_test = df_test[feature_names], df_test[[target_column]]

        # scale the two dataframes
        feature_scaler = StandardScaler()
        feature_scaler.fit(X_train)
        X_train = feature_scaler.transform(X_train)
        X_test = feature_scaler.transform(X_test)

        target_scaler = StandardScaler()
        target_scaler.fit(y_train)
        y_train = target_scaler.transform(y_train)
        y_test = target_scaler.transform(y_test)

        # fit and compute rmse and mae
        model.fit(X_train, y_train.ravel())
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1,1)

        rmse_fold = np.sqrt(mean_squared_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test)))
        rmse_list.append(rmse_fold)

        mae_fold = mean_absolute_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test))
        mae_list.append(mae_fold)

        r2_fold = r2_score(target_scaler.inverse_transform(y_test), target_scaler.inverse_transform(predictions))
        r2_list.append(r2_fold)


    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))
    r2 = np.mean(np.array(r2_list))

    return rmse, mae, r2

