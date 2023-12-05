import argparse


def get_args(cross_val=False):

    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", default=False, action="store_true")

    parser.add_argument(
        "--gpu",
        default=False,
        action="store_true",
        help="Availability of GPU",
    )
    parser.add_argument(
        "--transfer_learning",
        default=False,
        action="store_true",
        help="Re-trained a previous model",
    )
    parser.add_argument(
        "--random_state",
        default=0,
        action="store",
        type=int,
        help="random state to be selected for sampling/shuffling")
    parser.add_argument(
        "--batch-size",
        default=64,
        action="store",
        type=int,
        help="batch size while training the selectivity model"
    )
    parser.add_argument(
        "--max-epochs",
        default=100,
        action="store",
        type=int,
        help="number of epochs while training the selectivity model",
    )
    parser.add_argument(
        "--save_dir",
        default="results/cross_val_test/",
        type=str,
        help="path to the folder to save the predictions",
    )
    parser.add_argument(
        "--trained_dir",
        type=str,
        help="path to the checkpoint file of the trained model",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        help="path to the file for the predictions",
    )
    parser.add_argument(
        "--rxn_id_column",
        default="rxn_id",
        type=str,
        help="the column in which the rxn_id are stored",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to data",
    )
    parser.add_argument(
        "--splits",
        nargs=3,
        type=int,
        default=[80, 10, 10],
        help="split of the dataset into testing, validating, and training. The sum should be 100",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="the number of models to be ensembled",
    )
    parser.add_argument(
        "--train_valid_set_path",
        type=str,
        default=None,
        help="in case of selective sampling, indicates path to combined training and validation set csv-file",
    )
    parser.add_argument(
        "--test_set_path",
        type=str,
        default=None,
        help="in case of selective sampling, indicates path to the test set csv-file",
    )

    # Model args
    parser.add_argument(
        "--layers",
        default=0,
        action="store",
        type=int,
        help="number of hidden layers in the Feed Forward NN",
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        action="store",
        type=float,
        help="Dropout probability",
    )
    parser.add_argument(
        "--hidden-size",
        default=230,
        action="store",
        type=int,
        help="Dimensionality of hidden layers in MPN",)
    parser.add_argument(
        "--learning_rate",
        default=0.0277,
        action="store",
        type=float,
        help="a", )
    parser.add_argument(
        "--lr_ratio",
        default=0.95,
        action="store",
        type=float,
        help="a", )
    parser.add_argument(
        '--features',
        nargs="+",
        type=str,
        default=['dG_forward', 'dG_reverse', 'q_reac0', 'qH_reac0', 'q_reac1', 's_reac1', 'q_prod0',
                 's_prod0', 'q_prod1', 'qH_prod1', 'BV_reac1', 'BV_prod0', 'fr_dG_forward', 'fr_dG_reverse'],
        help='features for the different models')
    parser.add_argument(
        "--target_column",
        default="DG_TS_tunn",
        help="the column in which the activation energies are stored",
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        default=None,
        help='path to file containing the rxn-smiles',
    )

    # interactive way
    parser.add_argument("--mode", default='client', action="store", type=str)
    parser.add_argument("--host", default='127.0.0.1', action="store", type=str)
    parser.add_argument("--port", default=57546, action="store", type=int)

    if cross_val:
        parser.add_argument(
            "--k_fold",
            default=10,
            type=int,
            help="(Optional) # fold for cross-validation",
        )
        parser.add_argument(
            "--sample",
            type=int,
            help="(Optional) Randomly sample part of data for training during cross-validation",
        )

    return parser.parse_args()
