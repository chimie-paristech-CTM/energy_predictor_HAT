from rdkit import Chem
import numpy as np

def atom_map_dict(mol, includeHs = True):
    '''
    Get a dictionary of atom map numbers to atom indices. Determines the order of the predicted properties.
    
    params:
        mol: rdkit mol (already mapped atoms (based on mapped SMILES) ensure consistency with targets in case of training)
        includeHs: include hydrogens in dictionary (default: True)
    return: 
        fatom_index: dictionary (key: map nr.; value: rdkit atom index)
        mol: rdkit molecules including mapped Hs
    '''
    mol = Chem.AddHs(mol)
    current_map = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    consecutive = consecutive_nums([i for i in current_map if i != 0])

    if all([map == 0 for map in current_map]): # mol not mapped
            [a.SetAtomMapNum(a.GetIdx()+1) for a in mol.GetAtoms()]
    elif all([map != 0 for map in current_map]): # all atoms mapped (incl. explicit H); recommended for training
        s = sorted(list(set(current_map)))
        if min(current_map) != 1 or not consecutive: #if not 1st reactant in case of rxn SMILES or not mapped with consecutive no.; should not be the case for training!
            for a in mol.GetAtoms():
                a_map = a.GetAtomMapNum()
                if a_map != 0:
                    a.SetAtomMapNum(s.index(a_map)+1)
    elif any([map == 0 for map in current_map]): # explicit H not mapped
        if 1 not in current_map or not consecutive: # mapping does not start at 1 and/or does not have consecutive numbers
            s = sorted(list(set(current_map)))
            for a in mol.GetAtoms():
                a_map = a.GetAtomMapNum()
                if a_map != 0:
                    a.SetAtomMapNum(s.index(a_map))
        if includeHs:
            map_count = Chem.RemoveAllHs(mol).GetNumAtoms()
            interim_dict = {a.GetAtomMapNum() - 1: a.GetIdx() for a in Chem.RemoveHs(mol).GetAtoms()}
            for map_idx in sorted(interim_dict)[:(map_count+1)]:
                atom = mol.GetAtomWithIdx(interim_dict[map_idx])
                atom_nb = atom.GetNeighbors()
                for elem in atom_nb:
                    if elem.GetSymbol() == 'H':
                        if elem.GetAtomMapNum() == 0: # condition due to H incl. in smile as relevant for stereochemistry
                            elem.SetAtomMapNum(map_count+1)
                        map_count += 1
    
    if not includeHs:
        mol = Chem.RemoveHs(mol)
    fatom_index = {a.GetAtomMapNum() - 1: a.GetIdx() for a in mol.GetAtoms()}
    
    return fatom_index, mol

def bond_map_dict(mol):
    '''
    Get a dictionary of bonds based on the atom map numbers to the bond indices of the current rdkit mol.
    The order of these bond indices will determine the order or the predicted bond properties 
    (not used in training if bond dictionary is provided).

    params:
        mol: rdkit mol with mapped atoms
    return:
        fbond_index: dictionary (key: atom map numbers; value: rdkit bond index)
    '''
    fbond_index = {'{}-{}'.format(*sorted([b.GetBeginAtom().GetAtomMapNum() - 1,
                                          b.GetEndAtom().GetAtomMapNum() - 1])): b.GetIdx()
                   for b in mol.GetBonds()}
    return fbond_index


def consecutive_nums(nums):
    '''
    Determine if numbers in list are all consecutive (no gaps).

    :param nums: List of integers
    :return: Boolean describing if consecutive
    '''
    if len(nums) < 1:
        return False
    min_val = min(nums)
    max_val = max(nums)
    if max_val - min_val + 1 == len(nums):
        for i in range(len(nums)):
            if nums[i] < 0:
                j = -nums[i] - min_val
            else:
                j = nums[i] - min_val
                if nums[j] > 0:
                    nums[j] = -nums[j]
                else:
                    return False
        return True
    return False