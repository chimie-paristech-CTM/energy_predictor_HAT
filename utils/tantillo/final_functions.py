import pandas as pd
from rdkit import Chem


def add_pred_tantillo(train_file, test_file, pred_file):

    data = pd.read_csv(train_file)
    data_steroids = pd.read_csv(test_file)
    pred = pd.read_pickle(pred_file)

    spin_rad = []
    q_rad = []
    q_mol = []
    q_molH = []
    bdfe = []
    fr_bde = []
    bv = []

    for row in data.itertuples():
        mol_smiles, rad_smiles = row.rxn_smile.split('.')
        bdfe.append(pred.loc[pred['smiles'] == rad_smiles].dG.values[0])
        fr_bde.append(pred.loc[pred['smiles'] == rad_smiles].frozen_dG.values[0])
        bv.append(pred.loc[pred['smiles'] == rad_smiles].Buried_Vol.values[0])
        idx_rad = get_rad_index(rad_smiles)
        idx_mol, idx_molh = get_mol_index(rad_smiles, mol_smiles, idx_rad)
        spin_rad.append(pred.loc[pred['smiles'] == rad_smiles].spin_densities.values[0][idx_rad])
        q_mol.append(pred.loc[pred['smiles'] == mol_smiles].charges_all_atom.values[0][idx_mol])
        q_molH.append(pred.loc[pred['smiles'] == mol_smiles].charges_all_atom.values[0][idx_molh])
        q_rad.append(pred.loc[pred['smiles'] == rad_smiles].charges_all_atom.values[0][idx_rad])

    data['s_rad'] = spin_rad
    data['q_rad'] = q_rad
    data['q_mol'] = q_mol
    data['q_molH'] = q_molH
    data['Buried_Vol'] = bv
    data['BDFE'] = bdfe
    data['fr_BDE'] = fr_bde

    data.reset_index(inplace=True)
    data.rename(columns={'index': 'rxn_id'}, inplace=True)
    data.to_pickle('tmp/tantillo_data/input_tantillo.pkl')
    data.to_csv('tmp/tantillo_data/input_tantillo.csv')

    spin_rad = []
    q_rad = []
    bdfe = []
    fr_bde = []
    bv = []

    for row in data_steroids.itertuples():
        rad_smiles, mol_smiles = row.rxn_smile.split('.')
        bdfe.append(pred.loc[pred['smiles'] == rad_smiles].dG.values[0])
        fr_bde.append(pred.loc[pred['smiles'] == rad_smiles].frozen_dG.values[0])
        bv.append(pred.loc[pred['smiles'] == rad_smiles].Buried_Vol.values[0])
        idx_rad = get_rad_index(rad_smiles)
        spin_rad.append(pred.loc[pred['smiles'] == rad_smiles].spin_densities.values[0][idx_rad])
        q_rad.append(pred.loc[pred['smiles'] == rad_smiles].charges_all_atom.values[0][idx_rad])

    data_steroids['s_rad'] = spin_rad
    data_steroids['q_rad'] = q_rad
    data_steroids['Buried_Vol'] = bv
    data_steroids['BDFE'] = bdfe
    data_steroids['fr_BDE'] = fr_bde

    data_steroids.reset_index(inplace=True)
    data_steroids.rename(columns={'index': 'rxn_id'}, inplace=True)
    data_steroids.to_pickle('tmp/tantillo_data/input_steroids_tantillo.pkl')
    data_steroids.to_csv('tmp/tantillo_data/input_steroids_tantillo.csv')

    return None


def canonicalize_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def get_rad_index(smiles):
    """ Get the index of the radical atom"""

    mol = Chem.MolFromSmiles(smiles)

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() == 1:
            idx = atom.GetIdx()

    return idx


def get_mol_index(rad_smiles, mol_smiles, rad_idx):
    """ Get the index of the radical atom in the molecule and of the H"""

    os_mol = Chem.MolFromSmiles(rad_smiles)
    cs_mol = Chem.MolFromSmiles(mol_smiles)

    substructure = os_mol.GetSubstructMatch(cs_mol)

    if not substructure:
        Chem.Kekulize(cs_mol, clearAromaticFlags=True)
        Chem.Kekulize(os_mol, clearAromaticFlags=True)
        substructure = os_mol.GetSubstructMatch(cs_mol)

    mol_idx = substructure.index(rad_idx)

    cs_mol = Chem.AddHs(cs_mol)

    atom = [atom for atom in cs_mol.GetAtoms() if atom.GetIdx() == mol_idx][0]

    h_idx = [ngb.GetIdx() for ngb in atom.GetNeighbors() if ngb.GetSymbol() == 'H'][0]

    return mol_idx, h_idx

