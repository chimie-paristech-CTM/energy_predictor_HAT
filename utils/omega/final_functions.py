import pandas as pd
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

parser = ArgumentParser()
parser.add_argument('--input_file', type=str, default='data/input_omega_ffnn.pkl',
                    help='path to file containing the train set')
parser.add_argument('--exp_file', type=str, default='data/input_omega_exp_ffnn.pkl',
                    help='path to file containing the exp set')
parser.add_argument('--selectivity_file', type=str, default='data/input_omega_selectivity_ffnn.pkl',
                    help='path to file containing the exp set')
parser.add_argument('--features', nargs="+", type=str, default=['dG_forward', 'dG_reverse', 'q_reac0',
                                                                'qH_reac0', 'q_reac1', 's_reac1', 'q_prod0',
                                                                's_prod0', 'q_prod1', 'qH_prod1', 'BV_reac1',
                                                                'BV_prod0', 'fr_dG_forward', 'fr_dG_reverse'],
                    help='features for the different models')




def exp_data_corr(exp_path, pred_path, logger):

    df_exp = pd.read_csv(exp_path, index_col=0)
    df_pred = pd.read_csv(pred_path, index_col=0)
    # train a model, predict the DG_TS in bietti's dataset
    model = LinearRegression()
    X_train, y_train = df_exp[['gibbs_exp']], df_pred[['DG_TS_tunn']]
    model.fit(X_train, y_train)
    logger.info(f"Coefficient and intercept: {model.coef_}   {model.intercept_}")
    y_exp_corr = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_exp_corr)  
    rmse = np.sqrt(mean_squared_error(y_train, y_exp_corr))
    r2 = r2_score(y_exp_corr, y_train) 
    logger.info(f"RMSE, MAE and R^2: {rmse}   {mae}   {r2}")


def empirical_model():
    """
        Empirical model omega paper
            dG_act = alpha*dG_rxn*(1-d) + beta*dX^2 + delta
    """

    alpha = 0.41
    beta = - 0.20
    delta = 19.70

    df_train = pd.read_csv('data/train_original_omega.csv', index_col=0)
    df_train['dG_rxn'] = df_train['BDFE_sub'] - df_train['BDFE_cat']
    df_train['dX'] = df_train['x_sub'] - df_train['x_cat']
    df_train['d_houk'] = df_train['d'].apply(lambda x: 0.44 if x else 0)

    df_train['dG_act_pred'] = alpha * df_train['dG_rxn'] * (1 - df_train['d_houk']) + beta * (df_train['dX']**2) + delta

    mae_train = mean_absolute_error(df_train['gibbs'], df_train['dG_act_pred'])
    r2_train = r2_score(df_train['gibbs'], df_train['dG_act_pred'])
    rmse_train = np.sqrt(mean_squared_error(df_train['gibbs'], df_train['dG_act_pred']))

    df_test = pd.read_csv('data/test_original_omega.csv', index_col=0)
    df_test['dG_rxn'] = df_test['BDFE_sub'] - df_test['BDFE_cat']
    df_test['dX'] = df_test['x_sub'] - df_test['x_cat']
    df_test['d_houk'] = df_test['d'].apply(lambda x: 0.44 if x else 0)

    df_test['dG_act_pred'] = alpha * df_test['dG_rxn'] * (1 - df_test['d_houk']) + beta * (df_test['dX']**2) + delta

    mae_test = mean_absolute_error(df_test['gibbs'], df_test['dG_act_pred'])
    r2_test = r2_score(df_test['gibbs'], df_test['dG_act_pred'])
    rmse_test = np.sqrt(mean_squared_error(df_test['gibbs'], df_test['dG_act_pred']))


if __name__ == '__main__':
    # set up
    args = parser.parse_args()
    df = pd.read_pickle(args.input_file)
    df_train = df.iloc[:240]
    df_train = df_train.loc[df_train['DG_TS'] != 'FALSE']
    df_test = df.iloc[240:]
    df_exp = pd.read_pickle(args.exp_file)
    df_selectivity = pd.read_pickle(args.selectivity_file)

    features = args.features
    df = prepare_df(df, features)
    df_train = prepare_df(df_train, features)
    df_test = prepare_df(df_test, features)

    # exp
    #df_exp = pd.read_pickle('data/input_omega_exp_ffnn_1.pkl')
    #features += ['gibbs_exp']
    #df_exp = prepare_df(df_exp, features)
    #rmse, mae, r2, ev = cross_val(df_exp, LinearRegression(), 10, target_column='gibbs_exp')
    #logger.info(f'Experimental data of bietti')
    #logger.info(f'RMSE, MAE, R^2 and explained variance for 10 folds linear regression: {rmse} {mae} {r2} {ev}')
    #rmse, mae, r2, ev = cross_val(df_exp, RandomForestRegressor(), 10, target_column='gibbs_exp')
    #logger.info(f'RMSE, MAE, R^2 and explained variance for 10 folds RF: {rmse} {mae} {r2} {ev}')

    #exp corr