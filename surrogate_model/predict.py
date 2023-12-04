"""Loads a trained model checkpoint and makes predictions on a dataset."""
from utilities.parsing import parse_predict_args
from run.make_predictions import make_predictions
from utilities.utils_gen import load_args

import pandas as pd
from rdkit import Chem
import numpy as np
import time

def num_atoms_bonds(smiles, explicit_Hs):
    m = Chem.MolFromSmiles(smiles)

    if explicit_Hs:
        m = Chem.AddHs(m)

    return len(m.GetAtoms()), len(m.GetBonds())

if __name__ == '__main__':
    args = parse_predict_args()
    train_args = load_args(args.checkpoint_paths[0])

    if not hasattr(train_args, 'single_mol_tasks'): 
            train_args.single_mol_tasks = False

    test_df = pd.read_csv(args.test_path, index_col=0)
    smiles = test_df.smiles.tolist() 
    
    start = time.time() 
    test_preds, test_smiles = make_predictions(args, smiles=smiles)
    end = time.time()

    print('time:{}s'.format(end-start))

    train_a_targets = train_args.atom_targets
    train_b_targets = train_args.bond_targets
    train_m_targets = train_args.mol_targets
    if train_args.single_mol_tasks:
        train_m_targets= [item for sublist in train_m_targets for item in sublist]
    n_atoms, n_bonds = zip(*[num_atoms_bonds(x, train_args.explicit_Hs) for x in smiles]) 

    df = pd.DataFrame({'smiles': smiles})

    for idx, target in enumerate(train_a_targets):
        props = test_preds[idx]
        props = np.split(props.flatten(), np.cumsum(np.array(n_atoms)))[:-1]
        df[target] = props

    n_a_targets = len(train_a_targets)

    for idx, target in enumerate(train_b_targets):
        props = test_preds[idx+n_a_targets]
        props = np.split(props.flatten(), np.cumsum(np.array(n_bonds)))[:-1]
        df[target] = props

    n_ab_targets = len(train_a_targets) + len(train_b_targets)

    for idx, target in enumerate(train_m_targets):
        props = test_preds[idx+n_ab_targets]
        df[target] = props
    
    df.to_pickle(args.preds_path)
