import pandas as pd
from sklearn.linear_model import LinearRegression


def coef_inter(df, g_r='dG_rxn', g_a='dG_act_corrected'):

    X = df[[g_r]]
    y = df[[g_a]]

    model = LinearRegression()
    model.fit(X, y)

    return model.coef_, model.intercept_


def calc(x, slope, intercept):

    y = slope*x + intercept

    return y



if __name__ == '__main__':  

    df_dft = pd.read_csv('../../tmp/own_dataset/reactivity_database_corrected.csv', index_col=0)
    slope, intercept = coef_inter(df_dft)
    df_ml = pd.read_csv('../../tmp/input_ffnn.csv', index_col=0)
    df_ml['dG_act_eyring'] = df_ml['dG_rxn'].apply(lambda x: calc(x, slope[0], intercept[0])[0])
    df_ml['ddG_act'] = df_dft['dG_act_corrected'] - df_ml['dG_act_eyring']

    df_ml.to_csv('aa.csv')