import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def final_output(csv_pred, csv_input):

    df_pred = pd.read_csv(csv_pred, index_col=0)
    df_input = pd.read_csv(csv_input, index_col=0)

    df_pred['dG_rxn'] = df_input['dG_rxn']
    df_pred.to_csv('output.csv')

    return None


def read_log(log_file):

    with open(log_file, 'r') as file:
        lines = file.readlines()
    
    r2 = lines[-1].split()[-1]
    mae = lines[-2].split()[-1]
    rmse = lines[-3].split()[-1]

    return mae, rmse, r2


def get_stats(true, pred):

    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))

    return mae, rmse, r2

