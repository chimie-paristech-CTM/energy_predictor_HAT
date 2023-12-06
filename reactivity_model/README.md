# FFNN Property Prediction

The repository uses a feed forward network applied to a learned VB-representation to predict activation energies with tunneling corrections.

## Data
A model dataset has been included in the main directory. Data points are formatted as follows:


| | rxn_id | G_r  | DG_TS | DG_TS_tunn | dG_forward | dG_forward | 
|-|--------|------|-------|------------|------------|------------|
|0| 234077 | 7.30 | 29.83 | 27.72      | 85.49      | 75.62      |


## Training a model
A simple model can be trained by running the command:

```
python reactivity.py --data_path <path to reaction data .csv file> [--learning_rate <initial learning rate>] [--lr_ratio <learning rate decaying ratio>] 
[--layers <number of hidden layers>] [--random_state <random state to be selected for sampling/shuffling>] 
[--ensemble_size <the number of models to be ensembled>] [--splits <split of the dataset into testing, validating, and training. The sum should be 100>]
[--model_dir <path to the checkpoint file of the trained model>] [--hidden-size <Dimensionality of hidden layers in MPN>] 
```

As a pre-processing step, targets and descriptors are normalized; the scalers are saved in directories in the `save_dir` 
with name `scalers_{n_ensemble}`. Next, the training is performed and a checkpoint for every model in the ensemble is saved 
as `best_model_{n_ensemble}.ckpt` in the same directory.

For example:

```
python train.py --save_dir 'results/final_model_4/' --data_path 'desc/input_ffnn.pkl' --ensemble_size 4
```

It can also be trained with gpu by simply adding a gpu flag (with a small datset and small model, this is runnable on cpu):

```
python train.py --save_dir 'results/final_model_4/' --data_path 'desc/input_ffnn.pkl' --ensemble_size 4 --gpu
```

To view all arguments with argparse:

```
python train.py -h
```

## Making predictions
The trained model can be loaded directly from a TorchLightning checkpoint file and used to make predictions.

```
python predict.py --pred_file 'desc/input_ffnn_rmechdb_full_pred.csv' --trained_dir 'results/final_model_rmechdb_4_water/' --ensemble_size 4 --save_dir 'results/pred_rmechdb_full_data_4/' 
```

Again, this can also be made to run on gpu by adding `--gpu` at the end of the call.

## Cross-validating
To perform a cross-validation, run (additional parameters to finetune the model architecture have been discussed above):
```
python cross_val.py --data_path <path to reaction data .csv file> [--k_fold <number of folds>] 
```

For example:
```
python cross_val.py --data_path desc/input_ffnn.pkl --k_fold 10 --hidden-size 50 --learning_rate 0.0685 --lr_ratio 0.93 --random_state 2 --ensemble_size 10 --features 's_rad' --model_dir cross_val_ensemble10
```

## Trained Model
This repository contains a trained model here: `results/final_model_4/`. The model was trained on our in-house dataset of HAT reactions.

## Transfer Learning
To re-train a previous model with new data, add `--transfer_learning` and specify the `trained_dir` 

For example:
```
python3 train.py --save_dir 'results/final_model_rmechdb_4_water/' --data_path 'desc/input_ffnn_rmechdb_water.pkl' --ensemble_size 4 --transfer_learning --trained_dir '/resultsfinal_model_4/'
```