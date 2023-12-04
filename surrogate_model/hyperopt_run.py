"""Optimizes hyperparameters using Bayesian optimization.
Dependent files in hyperparam_opt folder adapted from chemprop: https://github.com/chemprop/chemprop
"""
from hyperparam_opt.hyperparameter_opt import chemprop_hyperopt

if __name__ == '__main__':
    chemprop_hyperopt()