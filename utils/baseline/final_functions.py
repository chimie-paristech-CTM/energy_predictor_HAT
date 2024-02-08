import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from utils.baseline.cross_val import cross_val


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



def get_cross_val_accuracy_rf_descriptors(df, logger, n_fold, split_dir=None, target_column='DG_TS'):
    """
    Get the random forest (descriptors) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = RandomForestRegressor(n_estimators=600, max_features=0.8, min_samples_leaf=1)
    rmse, mae, r2 = cross_val(df, model, n_fold, target_column=target_column, split_dir=split_dir)
    logger.info(f'{n_fold}-fold CV RMSE, MAE and R^2 for RF: {rmse} {mae} {r2}')