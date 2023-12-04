import pandas as pd
from argparse import ArgumentParser
from utils.log import create_logger
from utils.input_for_pred import create_input_pred
from utils.create_input_ffnn import create_input_ffnn
from utils.run_models import run_surrogate, run_reactivity
from utils.final_output import final_output
import time


parser = ArgumentParser()
parser.add_argument('--rxn_smiles', type=str, help='string of the reaction smiles')
parser.add_argument('--csv_file', type=str, help='path to file containing the rxn-smiles')

if __name__ == '__main__':

    initial_time = time.time()
    args = parser.parse_args()
    logger = create_logger()

    logger.info(f"reaction smiles: {args.rxn_smiles}")
    logger.info(f"csv file: {args.csv_file}")

    if not args.rxn_smiles and not args.csv_file:
        logger.info('Nothing to do :(')
    else:
        if args.rxn_smiles:
            rxns = [args.rxn_smiles]
            rxns_input = pd.DataFrame(rxns, columns=['RXN_SMILES'])
        else:
            rxns_input = pd.read_csv(args.csv_file, index_col=0)

        create_input_pred(rxns_input)
        run_surrogate()
        create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/reactivity_database_mapped.csv')
        run_reactivity()
        final_output('tmp/pred.csv', 'tmp/input_ffnn.csv')
        final_time = time.time()
        logger.info(f'results in output.csv')
        logger.info(f'time: {final_time - initial_time} seconds')







