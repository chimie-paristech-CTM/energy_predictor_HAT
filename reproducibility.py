import pandas as pd
from utils.log import create_logger
from utils.input_for_pred import create_input_pred
from utils.run_models import run_surrogate, run_cv
from utils.create_input_ffnn import create_input_ffnn

if __name__ == '__main__':

    # CV Hong data (DOI	https://doi.org/10.1039/D1QO01325D)
    logger = create_logger('hong_data.log')
    df_hong = pd.read_csv('tmp/training_hong_clean.csv', index_col=0)
    create_input_pred(df_hong, 'DG_TS')
    run_surrogate()
    logger.info('Surrogate model done')
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/reactivity_database_mapped.csv', 'DG_TS')
    run_cv('tmp/input_ffnn.pkl', 'DG_TS', 'tmp/cv_hong')
    logger.info('CV done')
    logger.info('Results in tmp/cv_hong/ffn_train.log')


