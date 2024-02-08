import pandas as pd
from rdkit import Chem


def create_input_pred(df, target_column=None):

    rxns = df['RXN_SMILES'].tolist()
    ids = df.index.values.tolist()
    rxns_mapped = []

    for rxn in rxns:
        rxns_mapped.append(map_rxn(rxn))

    df_mapped = pd.DataFrame()

    df_mapped['rxn_smiles'] = rxns_mapped
    df_mapped['rxn_id'] = ids
    if target_column:
        if target_column in df.columns:
            targets = df[target_column].tolist()
            df_mapped[target_column] = targets
    
    #df_mapped.to_csv('tmp/reactivity_database_mapped_before_filter.csv')
    df_mapped = df_mapped[df_mapped['rxn_smiles'] != False]
    df_mapped.reset_index(inplace=True)
    df_mapped.to_csv('tmp/reactivity_database_mapped.csv')

    smiles = []

    for row in df_mapped.itertuples():

        r1, r2, p1, p2 = reacs_prods(row.rxn_smiles)
        smiles.append(r1)
        smiles.append(r2)
        smiles.append(p1)
        smiles.append(p2)

    smiles = list(set(smiles))

    df_smiles = pd.DataFrame(smiles, columns=['smiles'])
    df_smiles.to_csv('tmp/species_reactivity_dataset.csv')


def map_rxn(rxn):

    reacs, prods = rxn.split('>>')

    smi_mol1, smi_rad2 = reacs.split('.')
    smi_rad1, smi_mol2 = prods.split('.')

    mol1 = Chem.MolFromSmiles(smi_mol1)
    rad2 = Chem.MolFromSmiles(smi_rad2)
    mol2 = Chem.MolFromSmiles(smi_mol2)
    rad1 = Chem.MolFromSmiles(smi_rad1)

    substruct1 = rad1.GetSubstructMatch(mol1)
    substruct2 = rad2.GetSubstructMatch(mol2)

    n_atoms_mol1 = mol1.GetNumAtoms()

    # Now we will have some problems because some radicals are not aromatic, so the command won't find any substructmatch

    sanitize1 = False
    sanitize2 = False

    if not substruct1:
        Chem.Kekulize(mol1, clearAromaticFlags=True)
        Chem.Kekulize(rad1, clearAromaticFlags=True)
        substruct1 = rad1.GetSubstructMatch(mol1)
        sanitize1 = True

    if not substruct2:
        Chem.Kekulize(mol2, clearAromaticFlags=True)
        Chem.Kekulize(rad2, clearAromaticFlags=True)
        substruct2 = rad2.GetSubstructMatch(mol2)
        sanitize2 = True

    if (not substruct2) or (not substruct1):
        return False

    # rxn:   mol1 + rad2 >> rad1 + mol2

    for idx, a_mol1 in enumerate(mol1.GetAtoms()):
        a_mol1.SetAtomMapNum(idx + 1)

    for idx, a_rad2 in enumerate(rad2.GetAtoms()):
        a_rad2.SetAtomMapNum(idx + 1 + n_atoms_mol1)

    for idx_mol1, idx_rad1 in enumerate(substruct1):
        atom_rad1 = rad1.GetAtomWithIdx(idx_rad1)
        atom_mol1 = mol1.GetAtomWithIdx(idx_mol1)
        map_number1 = atom_mol1.GetAtomMapNum()
        atom_rad1.SetAtomMapNum(map_number1)

    for idx_mol2, idx_rad2 in enumerate(substruct2):
        atom_rad2 = rad2.GetAtomWithIdx(idx_rad2)
        atom_mol2 = mol2.GetAtomWithIdx(idx_mol2)
        map_number2 = atom_rad2.GetAtomMapNum()
        atom_mol2.SetAtomMapNum(map_number2)

    # and now we have to "re-aromatize" the aromatic molecules or radicals"

    if sanitize1:
        Chem.SanitizeMol(mol1)
        Chem.SanitizeMol(rad1)

    if sanitize2:
        Chem.SanitizeMol(mol2)
        Chem.SanitizeMol(rad2)

    smi_map_mol1 = Chem.MolToSmiles(mol1)
    smi_map_rad2 = Chem.MolToSmiles(rad2)
    smi_map_mol2 = Chem.MolToSmiles(mol2)
    smi_map_rad1 = Chem.MolToSmiles(rad1)

    rxn_mapped = f"{smi_map_mol1}.{smi_map_rad2}>>{smi_map_rad1}.{smi_map_mol2}"

    return rxn_mapped


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

