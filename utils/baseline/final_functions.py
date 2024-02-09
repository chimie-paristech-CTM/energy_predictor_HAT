from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from utils.common import final_eval, cross_val



def get_cross_val_accuracy_ada_boost_regression(df, logger, n_fold, split_dir=None, target_column='DG_TS'):
    
    # ExtraTrees
    XTrees_R = ExtraTreesRegressor(
        n_estimators=30,
        #    n_estimators=100,
        criterion='squared_error',
        #    max_depth=50,
        min_samples_split=5,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=1,
        random_state=0,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None)

    ada = AdaBoostRegressor(base_estimator=XTrees_R,
                            n_estimators=50,
                            learning_rate=1.0,
                            loss='exponential',  # ‘linear’, ‘square’, ‘exponential’
                            random_state=None)
    
    rmse, mae, r2 = cross_val(df, ada, n_fold, target_column=target_column, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE, MAE and R^2 for AdaBoost: {rmse} {mae} {r2}')


def get_cross_val_accuracy_rf_descriptors(df, logger, n_fold, parameters, split_dir=None, target_column='DG_TS'):
    """
    Get the random forest (descriptors) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = RandomForestRegressor(n_estimators=int(parameters['n_estimators']),
            max_features=parameters['max_features'], min_samples_leaf=int(parameters['min_samples_leaf']))
    rmse, mae, r2 = cross_val(df, model, n_fold, target_column=target_column, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE, MAE and R^2 for RF: {rmse} {mae} {r2}')


def get_accuracy_linear_regression(df_train, df_test, logger, target_column, print_pred=False, name_out='pred'):

    model = LinearRegression()
    rmse, mae, r2 = final_eval(df_train, df_test, model, target_column, print_pred, name_out)

    logger.info(f'RMSE, MAE and R^2 for linear regression: {rmse} {mae} {r2}')


def get_accuracy_rf_descriptors(df_train, df_test, logger, parameters, target_column, print_pred=False, name_out='pred'):
    """
    Get the random forest (descriptors) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = RandomForestRegressor(n_estimators=int(parameters['n_estimators']),
            max_features=parameters['max_features'], min_samples_leaf=int(parameters['min_samples_leaf']))
    rmse, mae, r2 = final_eval(df_train, df_test, model, target_column, print_pred, name_out)
    logger.info(f'RMSE, MAE, and R^2 for RF: {rmse} {mae} {r2}')