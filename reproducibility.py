import pandas as pd
from utils.log import create_logger
from utils.input_for_pred import create_input_pred
from utils.run_models import run_surrogate, run_cv
from utils.create_input_ffnn import create_input_ffnn
from utils.final_output import read_log
from utils.baseline.final_functions import get_cross_val_accuracy_ada_boost_regression, get_cross_val_accuracy_rf_descriptors
from utils.tantillo.final_functions import prepare_df, get_accuracy_linear_regression, add_pred_tantillo

if __name__ == '__main__':

    # # cross-validation in-house HAT dataset
    # logger = create_logger('own_dataset.log')
    # df = pd.read_csv('tmp/own_dataset/reactivity_database_corrected.csv', index_col=0)
    # create_input_pred(df, 'dG_act_corrected')
    # run_surrogate()
    # logger.info('Surrogate model done')
    # create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/reactivity_database_mapped.csv', 'dG_act_corrected')
    
    # run_cv('tmp/input_ffnn.pkl', 'dG_act_corrected', 'tmp/cv_own_dataset', 10, 1, random_state=0)
    # logger.info('CV done')
    # logger.info('Results in tmp/cv_own_dataset/ffn_train.log')
    # mae, rmse, r2 = read_log('tmp/cv_own_dataset/ffn_train.log')
    # logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')

    # run_cv('tmp/input_ffnn.pkl', 'dG_act_corrected', 'tmp/cv_own_dataset_4', 10, 4, random_state=0)
    # logger.info('CV done')
    # logger.info('Results in tmp/cv_own_dataset_4/ffn_train.log')
    # mae, rmse, r2 = read_log('tmp/cv_own_dataset_4/ffn_train.log')
    # logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation and 4 ensembles: {rmse} {mae} {r2}')

    # tantillo dataset
    logger = create_logger('tantillo_data.log')
    features = ['s_rad', 'Buried_Vol']
    run_surrogate(test_file='tantillo_data/species_reactivity_tantillo_dataset.csv')
    add_pred_tantillo(train_file='tmp/tantillo_data/clean_data_tantillo.csv',
                      test_file='tmp/tantillo_data/clean_data_steroids_tantillo.csv', 
                      pred_file='tmp/preds_surrogate.pkl')
    df_train_tantillo = pd.read_pickle('tmp/tantillo_data/input_tantillo.pkl')
    df_test_tantillo = pd.read_pickle('tmp/tantillo_data/input_steroids_tantillo.pkl')
    df_train_tantillo = prepare_df(df_train_tantillo, features)
    df_test_tantillo = prepare_df(df_test_tantillo, features)
    get_accuracy_linear_regression(df_train_tantillo, df_test_tantillo, logger)

    # # cross-validation Hong data (DOI https://doi.org/10.1039/D1QO01325D)
    # logger = create_logger('hong_data.log')
    # df_hong = pd.read_csv('tmp/hong_data/training_hong_clean.csv', index_col=0)
    # create_input_pred(df_hong, 'DG_TS')
    # run_surrogate()
    # logger.info('Surrogate model done')
    # create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/reactivity_database_mapped.csv', 'DG_TS')
    
    # for sample in [25, 50, 100, 200, 300, 400]:
    #     save_dir = f"tmp/cv_hong_{sample}"
    #     run_cv(data_path='tmp/input_ffnn.pkl', target_column='DG_TS', save_dir=save_dir, k_fold=5, ensemble_size=1, sample=sample)
    #     logger.info(f'CV done with {sample} datapoints')
    #     logger.info(f'Results in {save_dir}/ffn_train.log')
    #     mae, rmse, r2 = read_log(f'{save_dir}/ffn_train.log')
    #     logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation and {sample} datapoints: {rmse} {mae} {r2}')

    #     save_dir = f"tmp/cv_hong_{sample}_4_TF"
    #     run_cv(data_path='tmp/input_ffnn.pkl', target_column='DG_TS', save_dir=save_dir, k_fold=5, ensemble_size=4, sample=sample, transfer_learning=True)
    #     logger.info(f'CV done with {sample} datapoints, 4 ensembles and transfer learning')
    #     logger.info(f'Results in {save_dir}/ffn_train.log')
    #     mae, rmse, r2 = read_log(f'{save_dir}/ffn_train.log')
    #     logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation, {sample} datapoints, 4 ensembles and transfer learning: {rmse} {mae} {r2}')

    # run_cv('tmp/input_ffnn.pkl', 'DG_TS', 'tmp/cv_hong', 5, 1)
    # logger.info('CV done')
    # logger.info('Results in tmp/cv_hong/ffn_train.log')
    # mae, rmse, r2 = read_log('tmp/cv_hong/ffn_train.log')
    # logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation and all datapoints: {rmse} {mae} {r2}')

    # run_cv('tmp/input_ffnn.pkl', 'DG_TS', 'tmp/cv_hong_4', 5, 4)
    # logger.info('CV done with 4 ensembles and transfer learning')
    # logger.info('Results in tmp/cv_hong_4/ffn_train.log')
    # mae, rmse, r2 = read_log('tmp/cv_hong_4/ffn_train.log')
    # logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation, all datapoints, 4 ensembles and transfer learning:: {rmse} {mae} {r2}')

    # df_hong_desc = pd.read_pickle('tmp/input_ffnn.pkl')
    # df_hong = pd.read_csv('tmp/reactivity_database_mapped.csv')
    # df_hong_original = pd.read_csv('tmp/hong_data/TrainingSet-2926-PhysOrg.csv')

    # df_hong_original.drop(index=[0,1], axis=0, inplace=True)
    # df_hong_original.reset_index(inplace=True)
    # df_hong_original['index'] = df_hong_original['index'].apply(lambda x: x-2)
    # df_hong_original.rename(columns={'index': 'rxn_id'}, inplace=True)

    # df_hong_intersection = df_hong_original.loc[df_hong_original.index.isin(df_hong['rxn_id'])]
    # logger.info(f'======== 5-fold CV with RF and AdaBoost ========')
    # for sample in [25, 50, 100, 200, 300, 400]:
    #     logger.info(f'Datapoints: {sample}')
    #     split_dir = f"tmp/cv_hong_{sample}/splits"
    #     logger.info(f'AdaBoost with 50 descriptors')
    #     get_cross_val_accuracy_ada_boost_regression(df=df_hong_intersection, logger=logger, n_fold=5, split_dir=split_dir, target_column='Barrier')
    #     logger.info(f'Model with a learned-VB representation')
    #     get_cross_val_accuracy_rf_descriptors(df=df_hong_desc, logger=logger, n_fold=5, split_dir=split_dir, target_column='DG_TS')
    # logger.info(f'AdaBoost with 50 descriptors')
    # get_cross_val_accuracy_ada_boost_regression(df=df_hong_intersection, logger=logger, n_fold=5, split_dir='tmp/cv_hong/splits', target_column='Barrier')
    # logger.info(f'Model with a learned-VB representation')
    # get_cross_val_accuracy_rf_descriptors(df=df_hong_desc, logger=logger, n_fold=5, split_dir='tmp/cv_hong/splits', target_column='DG_TS')


