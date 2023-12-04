import pandas as pd


def final_output(csv_pred, csv_input):

    df_pred = pd.read_csv(csv_pred, index_col=0)
    df_input = pd.read_csv(csv_input, index_col=0)

    df_pred['dG_rxn'] = df_input['dG_rxn']
    df_pred.to_csv('output.csv')

    return None
