import pandas as pd
from utils.log import create_logger
from utils.input_for_pred import create_input_pred
from utils.run_models import run_surrogate, run_cv, run_train, run_reactivity
from utils.create_input_ffnn import create_input_ffnn
from utils.final_output import read_log, get_stats
from utils.baseline.final_functions import get_cross_val_accuracy_ada_boost_regression, get_cross_val_accuracy_rf_descriptors
from utils.baseline.final_functions import get_accuracy_linear_regression, get_accuracy_rf_descriptors
from utils.tantillo.final_functions import add_pred_tantillo
from utils.omega.final_functions import exp_data_corr
from utils.common import prepare_df

if __name__ == '__main__':

    # # cross-validation in-house HAT dataset
    logger = create_logger('own_dataset.log')
    logger.info('********************************')
    logger.info('======= In-HOUSE DATASET =======')
    df = pd.read_csv('tmp/own_dataset/reactivity_database_corrected.csv', index_col=0)
    create_input_pred(df=df, target_column='dG_act_corrected', target_column_2='dG_rxn')
    run_surrogate()
    logger.info('Surrogate model done')
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/reactivity_database_mapped.csv', 'dG_act_corrected')

    run_cv('tmp/input_ffnn.pkl', 'dG_act_corrected', 'tmp/cv_own_dataset', 10, 1, random_state=0)
    logger.info('CV done')
    logger.info('Results in tmp/cv_own_dataset/ffn_train.log')
    mae, rmse, r2 = read_log('tmp/cv_own_dataset/ffn_train.log')
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')

    run_cv('tmp/input_ffnn.pkl', 'dG_act_corrected', 'tmp/cv_own_dataset_4', 10, 4, random_state=0)
    logger.info('CV done')
    logger.info('Results in tmp/cv_own_dataset_4/ffn_train.log')
    mae, rmse, r2 = read_log('tmp/cv_own_dataset_4/ffn_train.log')
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation and 4 ensembles: {rmse} {mae} {r2}')

    # tantillo dataset (DOI: https://doi.org/10.1002/cmtd.202100108)
    logger = create_logger('tantillo_data.log')
    logger.info('********************************')
    logger.info('======= TANTILLO DATASET =======')
    features = ['s_rad', 'Buried_Vol']
    run_surrogate(test_file='tmp/tantillo_data/species_reactivity_tantillo_dataset.csv')
    logger.info('Surrogate model done')
    add_pred_tantillo(train_file='tmp/tantillo_data/clean_data_tantillo.csv',
                      test_file='tmp/tantillo_data/clean_data_steroids_tantillo.csv', 
                      pred_file='tmp/preds_surrogate.pkl')
    df_train_tantillo = pd.read_pickle('tmp/tantillo_data/input_tantillo.pkl')
    df_test_tantillo = pd.read_pickle('tmp/tantillo_data/input_steroids_tantillo.pkl')
    df_train_tantillo = prepare_df(df_train_tantillo, features)
    df_test_tantillo = prepare_df(df_test_tantillo, features)
    get_accuracy_linear_regression(df_train_tantillo, df_test_tantillo, logger, 'DFT_Barrier')

    # omega dataset (DOI: https://doi.org/10.1021/acsomega.2c03252)
    logger = create_logger('omega_data.log')
    logger.info('********************************')
    logger.info('======= OMEGA DATASET =======')
    features = ['dG_forward', 'dG_reverse', 'q_reac0', 'qH_reac0', 'q_reac1', 's_reac1', 'q_prod0', 's_prod0', 'q_prod1', 'qH_prod1', 'BV_reac1', 'BV_prod0', 'fr_dG_forward', 'fr_dG_reverse']
    run_surrogate(test_file='tmp/omega_data/species_reactivity_omega_dataset.csv')
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/omega_data/clean_data_omega.csv', 'G_act', 'tmp/omega_data/additional_data_omega.pkl')
    df_omega = pd.read_pickle('tmp/input_ffnn.pkl')
    df_train = df_omega.iloc[:240]
    df_train = df_train.loc[df_train['G_act'] != 'FALSE']
    df_test = df_omega.iloc[240:]
    df_train.to_pickle('tmp/input_ffnn.pkl')
    df_train = prepare_df(df_train, features)
    df_test = prepare_df(df_test, features)
    logger.info('Train set')
    get_accuracy_linear_regression(df_train, df_train, logger, 'G_act')
    logger.info('Test set')
    get_accuracy_linear_regression(df_train, df_test, logger, 'G_act')
    parameters_rf = {'n_estimators': 300, 'max_features': 1, 'min_samples_leaf': 1}
    get_accuracy_rf_descriptors(df_train, df_test, logger, parameters_rf, 'G_act')
    df_train.to_csv('tmp/omega_data/train_valid_set.csv')
    df_test.to_csv('tmp/omega_data/test_set.csv')
    run_cv(target_column='G_act', save_dir='tmp/cv_omega_TF_4', k_fold=10, ensemble_size=4, transfer_learning=True, test_set='tmp/omega_data/test_set.csv', train_valid_set='tmp/omega_data/train_valid_set.csv', random_state=0, batch_size=24)
    mae, rmse, r2 = read_log('tmp/cv_omega_TF_4/ffn_train.log')
    logger.info("4 ensembles and transfer learning")
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    run_cv(target_column='G_act', save_dir='tmp/cv_omega_no_TF_4', k_fold=10, ensemble_size=4, test_set='tmp/omega_data/test_set.csv', train_valid_set='tmp/omega_data/train_valid_set.csv', random_state=0, batch_size=24)
    mae, rmse, r2 = read_log('tmp/cv_omega_no_TF_4/ffn_train.log')
    logger.info("4 ensembles")
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    run_cv(target_column='G_act', save_dir='tmp/cv_omega_no_TF_1', k_fold=10, ensemble_size=1, test_set='tmp/omega_data/test_set.csv', train_valid_set='tmp/omega_data/train_valid_set.csv', random_state=0, batch_size=24)
    mae, rmse, r2 = read_log('tmp/cv_omega_no_TF_1/ffn_train.log')
    logger.info("1 ensemble")
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    run_cv(target_column='G_act', save_dir='tmp/cv_omega_TF_1', k_fold=10, ensemble_size=1, transfer_learning=True, test_set='tmp/omega_data/test_set.csv', train_valid_set='tmp/omega_data/train_valid_set.csv', random_state=0, trained_dir='reactivity_model/results/final_model_1/', batch_size=24)
    mae, rmse, r2 = read_log('tmp/cv_omega_TF_1/ffn_train.log')
    logger.info("1 ensemble and transfer learning")
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')

    run_train(save_dir='tmp/final_model_omega_4', data_path='tmp/input_ffnn.pkl', target_column='G_act', batch_size=24)
    run_train(save_dir='tmp/final_model_omega_1', data_path='tmp/input_ffnn.pkl', target_column='G_act', batch_size=24, ensemble_size=1, trained_dir='reactivity_model/results/final_model_1/')

    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/omega_data/clean_data_omega_selectivity.csv', 'gibbs', 'tmp/omega_data/additional_data_omega.pkl', 'input_ffnn_selectivity')
    df_selectivity = pd.read_pickle('tmp/input_ffnn_selectivity.pkl')
    df_selectivity.rename(columns={'gibbs': 'G_act'}, inplace=True)
    df_selectivity = prepare_df(df_selectivity, features)
    get_accuracy_linear_regression(df_train, df_selectivity, logger, 'G_act', print_pred=True, name_out='pred_selectivity_lm')
    get_accuracy_rf_descriptors(df_train, df_selectivity, logger, parameters_rf, 'G_act', print_pred=True, name_out='pred_selectivity_rf')
    df_selectivity.to_pickle('tmp/input_ffnn_selectivity.pkl')
    df_selectivity.to_csv('tmp/input_ffnn_selectivity.csv')
    run_reactivity(trained_dir='tmp/final_model_omega_4/', target_column='G_act', ensemble_size=4, input_file='input_ffnn_selectivity.csv')
    df_selectivity_pred = pd.read_csv('tmp/pred.csv')
    mae, rmse, r2 = get_stats(df_selectivity['G_act'], df_selectivity_pred['DG_TS_tunn'])
    logger.info("Selectivity 4 ensemble")
    logger.info(f'RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')

    run_reactivity(trained_dir='tmp/final_model_omega_1/', target_column='G_act', ensemble_size=1, input_file='input_ffnn_selectivity.csv')
    df_selectivity_pred = pd.read_csv('tmp/pred.csv')
    mae, rmse, r2 = get_stats(df_selectivity['G_act'], df_selectivity_pred['DG_TS_tunn'])
    logger.info("Selectivity 1 ensemble")
    logger.info(f'RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')

    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/omega_data/clean_data_omega_exp.csv', 'gibbs_exp', 'tmp/omega_data/additional_data_omega.pkl', output='input_ffnn_bietti')
    
    # cross-validation Hong data (DOI https://doi.org/10.1039/D1QO01325D)
    logger = create_logger('hong_data.log')
    logger.info('********************************')
    logger.info('======= HONG DATASET =======')
    df_hong = pd.read_csv('tmp/hong_data/training_hong_clean.csv', index_col=0)
    create_input_pred(df_hong, 'DG_TS')
    run_surrogate()
    logger.info('Surrogate model done')
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/reactivity_database_mapped.csv', 'DG_TS')
    
    for sample in [25, 50, 100, 200, 300, 400]:
        save_dir = f"tmp/cv_hong_{sample}"
        run_cv(data_path='tmp/input_ffnn.pkl', target_column='DG_TS', save_dir=save_dir, k_fold=5, ensemble_size=1, sample=sample, batch_size=32)
        logger.info(f'CV done with {sample} datapoints')
        logger.info(f'Results in {save_dir}/ffn_train.log')
        mae, rmse, r2 = read_log(f'{save_dir}/ffn_train.log')
        logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation and {sample} datapoints: {rmse} {mae} {r2}')

        save_dir = f"tmp/cv_hong_{sample}_4_TF"
        run_cv(data_path='tmp/input_ffnn.pkl', target_column='DG_TS', save_dir=save_dir, k_fold=5, ensemble_size=4, sample=sample, transfer_learning=True, batch_size=32)
        logger.info(f'CV done with {sample} datapoints, 4 ensembles and transfer learning')
        logger.info(f'Results in {save_dir}/ffn_train.log')
        mae, rmse, r2 = read_log(f'{save_dir}/ffn_train.log')
        logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation, {sample} datapoints, 4 ensembles and transfer learning: {rmse} {mae} {r2}')

    run_cv('tmp/input_ffnn.pkl', 'DG_TS', 'tmp/cv_hong', 5, 1, batch_size=32)
    logger.info('CV done')
    logger.info('Results in tmp/cv_hong/ffn_train.log')
    mae, rmse, r2 = read_log('tmp/cv_hong/ffn_train.log')
    logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation and all datapoints: {rmse} {mae} {r2}')

    run_cv('tmp/input_ffnn.pkl', 'DG_TS', 'tmp/cv_hong_4', 5, 4, batch_size=32)
    logger.info('CV done with 4 ensembles and transfer learning')
    logger.info('Results in tmp/cv_hong_4/ffn_train.log')
    mae, rmse, r2 = read_log('tmp/cv_hong_4/ffn_train.log')
    logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation, all datapoints, 4 ensembles and transfer learning:: {rmse} {mae} {r2}')

    df_hong_desc = pd.read_pickle('tmp/input_ffnn.pkl')
    df_hong = pd.read_csv('tmp/reactivity_database_mapped.csv')
    df_hong_original = pd.read_csv('tmp/hong_data/TrainingSet-2926-PhysOrg.csv')

    df_hong_original.drop(index=[0,1], axis=0, inplace=True)
    df_hong_original.reset_index(inplace=True)
    df_hong_original['index'] = df_hong_original['index'].apply(lambda x: x-2)
    df_hong_original.rename(columns={'index': 'rxn_id'}, inplace=True)

    df_hong_intersection = df_hong_original.loc[df_hong_original.index.isin(df_hong['rxn_id'])]
    logger.info(f'======== 5-fold CV with RF and AdaBoost ========')
    parameters_rf = {'n_estimators': 300, 'max_features': 0.9, 'min_samples_leaf': 1}
    for sample in [25, 50, 100, 200, 300, 400]:
        logger.info(f'Datapoints: {sample}')
        split_dir = f"tmp/cv_hong_{sample}/splits"
        logger.info(f'AdaBoost with 50 descriptors')
        get_cross_val_accuracy_ada_boost_regression(df=df_hong_intersection, logger=logger, n_fold=5, split_dir=split_dir, target_column='Barrier')
        logger.info(f'Model with a learned-VB representation')
        get_cross_val_accuracy_rf_descriptors(df=df_hong_desc, logger=logger, n_fold=5, parameters=parameters_rf, split_dir=split_dir, target_column='DG_TS')
    logger.info(f'AdaBoost with 50 descriptors')
    get_cross_val_accuracy_ada_boost_regression(df=df_hong_intersection, logger=logger, n_fold=5, split_dir='tmp/cv_hong/splits', target_column='Barrier')
    logger.info(f'Model with a learned-VB representation')
    get_cross_val_accuracy_rf_descriptors(df=df_hong_desc, logger=logger, n_fold=5, parameters=parameters_rf, split_dir='tmp/cv_hong/splits', target_column='DG_TS')

    run_train(save_dir='tmp/final_model_hong_4', data_path='tmp/input_ffnn.pkl', target_column='DG_TS', batch_size=32)
    run_train(save_dir='tmp/final_model_hong_1', data_path='tmp/input_ffnn.pkl', target_column='DG_TS', batch_size=32, ensemble_size=1, trained_dir='reactivity_model/results/final_model_1/')

    run_cv(data_path='tmp/input_ffnn_bietti.pkl', target_column='gibbs_exp', save_dir='tmp/cv_hong_bietti_4', transfer_learning=True, batch_size=5, trained_dir='tmp/final_model_hong_4/', random_state=0)
    mae, rmse, r2 = read_log('tmp/cv_hong_bietti_4/ffn_train.log')
    logger.info("4 ensembles and transfer learning on Bietti Set")
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    run_cv(data_path='tmp/input_ffnn_bietti.pkl', target_column='gibbs_exp', save_dir='tmp/cv_hong_bietti_1', transfer_learning=True, batch_size=5, ensemble_size=1, trained_dir='tmp/final_model_hong_1/', random_state=0)
    mae, rmse, r2 = read_log('tmp/cv_hong_bietti_1/ffn_train.log')
    logger.info("1 ensemble and transfer learning on Bietti Set")
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')

    df_bietti = pd.read_pickle('tmp/input_ffnn_bietti.pkl')
    df_bietti.rename(columns={'gibbs_exp':'DG_TS'}, inplace=True)
    df_bietti.to_csv('tmp/input_ffnn_bietti.csv')
    run_reactivity(trained_dir='tmp/final_model_hong_4', target_column='DG_TS', input_file='input_ffnn_bietti.csv')
    df_bietti_pred = pd.read_csv('tmp/pred.csv')
    df_bietti = pd.read_csv('tmp/omega_data/clean_data_omega_exp.csv')
    exp_data_corr(exp_path='tmp/omega_data/clean_data_omega_exp.csv', pred_path='tmp/pred.csv', logger=logger)
    run_reactivity(trained_dir='tmp/final_model_hong_1', target_column='DG_TS', input_file='input_ffnn_bietti.csv', ensemble_size=1)
    df_bietti_pred = pd.read_csv('tmp/pred.csv')
    df_bietti = pd.read_csv('tmp/omega_data/clean_data_omega_exp.csv')
    exp_data_corr(exp_path='tmp/omega_data/clean_data_omega_exp.csv', pred_path='tmp/pred.csv', logger=logger)
    
