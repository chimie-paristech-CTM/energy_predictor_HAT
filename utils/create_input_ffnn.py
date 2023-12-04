#!/usr/bin/python
import pandas as pd
from rdkit import Chem

def create_input_ffnn(pkl_file, csv_file, additional_data=None, output='input_ffnn'):
    """ we're going to take the csv file of the baseline model and add the line of descriptors pred by the surrogate
    model this surrogate model now predict the charges of open and closed shell molecules and also the dG

    pkl_file (str) : path to file containing output surrogate model

    csv_file (str) : path to file containing input to baseline model

    """

    surrogate_data = pd.read_pickle(pkl_file)
    reactivity_data = pd.read_csv(csv_file, index_col=0)

    if additional_data:
        surrogate_additional = pd.read_pickle(additional_data)
        surrogate_data = pd.concat([surrogate_data, surrogate_additional], axis=0, ignore_index=True)
        surrogate_data.drop_duplicates(subset=['smiles'], keep="last", inplace=True)

    dG = []
    dG_forward = []
    dG_reverse = []
    fr_dG_forward = []
    fr_dG_reverse = []
    fr_dG = []
    spin_p1 = []
    charge_r2 = []
    spin_r2 = []
    chargeH_r1 = []
    chargeH_p2 = []
    charge_r1 = []
    charge_p2 = []
    charge_p1 = []
    BV_r1 = []
    BV_p0 = []

    for row in reactivity_data.itertuples():
        mol1, rad1, rad2, mol2 = reacs_prods(row.rxn_smiles)
        mol1_info = surrogate_data.loc[surrogate_data.smiles == mol1]
        rad1_info = surrogate_data.loc[surrogate_data.smiles == rad1]
        rad2_info = surrogate_data.loc[surrogate_data.smiles == rad2]
        mol2_info = surrogate_data.loc[surrogate_data.smiles == mol2]

        dG.append(rad2_info.dG.values[0] - rad1_info.dG.values[0])
        dG_forward.append(rad2_info.dG.values[0])
        dG_reverse.append(rad1_info.dG.values[0])
        fr_dG_forward.append(rad2_info.frozen_dG.values[0])
        fr_dG_reverse.append(rad1_info.frozen_dG.values[0])
        fr_dG.append(rad2_info.frozen_dG.values[0] - rad1_info.frozen_dG.values[0])
        BV_r1.append(rad1_info.Buried_Vol.values[0])
        BV_p0.append(rad2_info.Buried_Vol.values[0])

        rad1_idx = get_rad_index(rad1)
        rad2_idx = get_rad_index(rad2)
        spin_p1.append(rad2_info.spin_densities.values[0][rad2_idx])
        spin_r2.append(rad1_info.spin_densities.values[0][rad1_idx])
        charge_p1.append(rad2_info.charges_all_atom.values[0][rad2_idx])
        charge_r2.append(rad1_info.charges_all_atom.values[0][rad1_idx])

        atm_rad_idx, h_idx = get_mol_index(rad1, mol2, rad1_idx)
        charge_p2.append(mol2_info.charges_all_atom.values[0][atm_rad_idx])
        chargeH_p2.append(mol2_info.charges_all_atom.values[0][h_idx])

        atm_rad_idx, h_idx = get_mol_index(rad2, mol1, rad2_idx)
        charge_r1.append(mol1_info.charges_all_atom.values[0][atm_rad_idx])
        chargeH_r1.append(mol1_info.charges_all_atom.values[0][h_idx])

    input_ffnn = pd.DataFrame()

    input_ffnn['rxn_id'] = reactivity_data.index
    input_ffnn['dG_rxn'] = dG
    input_ffnn['dG_forward'] = dG_forward
    input_ffnn['dG_reverse'] = dG_reverse
    input_ffnn['fr_dG_rxn'] = fr_dG
    input_ffnn['fr_dG_forward'] = fr_dG_forward
    input_ffnn['fr_dG_reverse'] = fr_dG_reverse
    input_ffnn['s_prod0'] = spin_p1
    input_ffnn['q_reac1'] = charge_r2
    input_ffnn['s_reac1'] = spin_r2
    input_ffnn['qH_reac0'] = chargeH_r1
    input_ffnn['qH_prod1'] = chargeH_p2
    input_ffnn['q_reac0'] = charge_r1
    input_ffnn['q_prod1'] = charge_p2
    input_ffnn['q_prod0'] = charge_p1
    input_ffnn['BV_reac1'] = BV_r1
    input_ffnn['BV_prod0'] = BV_p0

    input_ffnn.to_csv(f'tmp/{output}.csv')
    input_ffnn.to_pickle(f'tmp/{output}.pkl')


def reacs_prods(smiles_rxn):
    """ Return the components of the rxn """

    dot_index = smiles_rxn.index('.')
    sdot_index = smiles_rxn.index('.', dot_index + 1)
    limit_index = smiles_rxn.index('>>')
    reac1 = smiles_rxn[:dot_index]
    reac2 = smiles_rxn[dot_index + 1: limit_index]
    prod1 = smiles_rxn[limit_index + 2: sdot_index]
    prod2 = smiles_rxn[sdot_index + 1:]

    return reac1, reac2, prod1, prod2


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
