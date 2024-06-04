# Activation energy with tunneling correction prediction with a learned VB-representation and on-the-fly quantum mechanical descriptors

[![ChemRxiv](https://img.shields.io/badge/arXiv-2312.13136-b31b1b.svg)](https://doi.org/10.26434/chemrxiv-2023-2n281)
[![DOI](http://img.shields.io/badge/DOI-10.1039/D4DD00043A-008000.svg)](https://doi.org/10.1039/D4DD00043A)

This repository contains the code for a fast prediction of activation energies. Code is provided "as-is". Minor edits may be required to tailor the scripts for different computational systems. 
The image below shows a schematic representation of the pipeline. 

![](toc.png)

### Conda environment

To set up a conda environment:

```
conda env create -f environment.yml
```

## Making predictions
As simple as:

```
python run.py --rxn_smiles 'CCO.C[CH2]>>CC[O].CC'
```

or

```
python run.py --csv_file tmp/examples.csv
```

The reaction smiles should be in the form of:

```
mol_1 + rad_2 >> rad_1 + mol_1
```

The first step is the prediction of relevant chemical information of reactants and products with the surrogate model, and a
learned VB-representation of the reaction smiles is generated. With this, the (tunneling corrected) activation energy is predicted. With the 
combination of both models, a full reaction profile can be generated quickly and accurately.

## Individual models
In the `reactivity_model` and `surrogate_model` directories you can find each individual model. In both folders, there is also a README in case you want 
to use just one part of the pipeline.

## Reproducibility
We provide a script `reproducibility.py` to generate the main results shown in the publication. Be aware that the values concerning the Random Forest can vary in a small range. To run this script, execute:

```python
python reproducibility.py
```

## Citation
If you use this code, please cite:

```
@article{hat_predictor,
         title={Repurposing QM Descriptor Datasets for on the Fly Generation of Informative Reaction Representations: 
         Application to Hydrogen Atom Transfer Reactions}, 
         author={Javier E. Alfonso-Ramos, Rebecca M. Neeser and Thijs Stuyver},
         journal="{Digital Discovery}",
         year="{2024}",
         volume="{3}",
         issue="{5}",
         pages="{919-931}",
         doi="10.1039/D4DD00043A",
         url="https://doi.org/10.1039/D4DD00043A"
}
```
