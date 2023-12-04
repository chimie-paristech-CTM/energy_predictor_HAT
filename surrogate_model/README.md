# Atomic and molecular property prediction for QM-augmented reactivity prediction
This folder contains a message passing neural network for the prediction of atomic and molecular properties. The code is adapted from [chemprop-atom-bond](https://github.com/yanfeiguan/chemprop-atom-bond) by [@yanfeiguan](https://github.com/yanfeiguan). 

## Training
To train a model, run:

```
python train.py --data_path <path> --save_dir <dir>
```

Where ```<path>``` is the path to a pickle file containing the training data and ```<dir>``` is the path to a directory where all checkpoints and predictions will be saved. For a description of the training data see section "Data".

Further important commands for training:
* ```--depth```: number of message-passing steps
* ```--hidden_size```: number of hidden nodes in the MPNN
* ```--ffn_hidden_size```: number of hidden nodes in all FFNNs
* ```--ffn_num_layers```: number of FFNN layers (including output)
* ```--atom_targets```: list of strings corresponding to column names for atom targets
* ```--bond_targets```: list of strings corresponding to column names for bond targets
* ```--mol_ext_targets```: list of strings corresponding to column names for extensive molecular targets
* ```--mol_int_targets```: list of strings corresponding to column names for intensive molecular targets
* ```--epochs```
* ```--batch_size```
* learning rate settings: ```--final_lr``` and ```--init_lr``` when using a SINEXP lr schedule (default for command ```--lr_schedule```)
* ```--explicit_Hs```: predict explicit Hydrogens (needs to match target data)
* ```--no_features_scaling```: turn off feature scaling (relevant if provide additional features with ```--features_path```)
* ```--target_scaling```: turn on target scaling (recommended)
* ```--target_scaler_type```: select trarget scaler type, default ```StandardScaler``` (alternative: ```MinMaxScaler```)
* ```--atom_wise_scaling```: 0 for global scaling, 1 for element-level scaling (must have same order as atom target list)
* ```--no_atom_scaling```: turn off atom-wise scaling
* ```-atom_constraints```: 0 if no constraint, 1 for charge constraint (same order as atom targets)
* ```--equal_lss_weights```: set all loss weights to 1; provide weights with ```--loss_weights```(floats in same order as targets)
* ```--seed```: random seed for splitting data
* ```--split_sizes```: train, val, test size (floats)
* ```--split_type```: select way to split data (choices: ```random```(default), ```predetermined, corssval, index_predetermined, train_eq-test```); check ```qm_pred/utilities/utils_data.py``` for different options
* ```--folds_file```: pickle file with indices for the test set subdivided into folds (only relevant if ```--split_type```=predetermined)
* ```--test_fold_index```: select which fold to use for the test set (only relevant if ```--split_type```=predetermined)
* ```--no_cache```: turn off caching mol2graph computation (recommended for big data files)
* ```--single_mol_tasks```: branch into individual FFNNs for all properties, otherwise extensive and intensive molecular properties will share a FFNN each

Consult ```qm_pred/utilities/parsing.py``` for more options.

## Predicting
To predict all QM properties with a pre-trained model, run:

```
python predict.py --test_path <path> --checkpoint_path <model_path> --preds_path <output>
```

Where ```<path>``` corresponds to a csv file containing atom-mapped SMILES to predict on (column header ```smiles```), ```<model_path>``` points to the model.py file of a trained model and ```<output>``` refers to the pickle file to save the predictions in.

### Trained Model
This repository contains a trained model here: ```qm_pred/qmdesc_wrap/model.pt```. This checkpoin was trained to predict atomic (Mulliken partial charges, Mulliken spin densities) and molecular (Buried Volume, Bond Dissociation Free Energy, frozen Bond Dissociation Energy) properties.
The model was trained on an dataset of QM properties for 40,000 closed-shell organic molecules and 200,000 radical analogs by Paton and co-workers, BDE-db ([St. John et al](https://doi.org/10.1038/s41597-020-00588-x)).

## Data
The training data consists of the following columns: ```smiles```recording all atom-mapped SMILES strings, all QM properties individually with all atom and molecular proeprties per molecule in one row stored in lists, ```CHEMBL_ID``` representing any compound ID (optional: if available, add command ```--use_compound_names```, ```CONF_ID```representing the conformer ID (optional: if availabel, add command ```--use_conf_id```). The index is stored in the first column.
The dataframe may look like this:

| |smiles|dG|spin_densities|
|-|------|--|--------------|
|0|[O:1][H:2]|[109.00]|[1.025, -0.025]|

It is recommended to generate the pickle file using this repository's conda environment to avoid version issues with the pandas package.

