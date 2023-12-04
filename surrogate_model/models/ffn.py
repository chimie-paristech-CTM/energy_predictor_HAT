from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import re
import numpy as np

from utilities.utils_nn import get_activation_function

class AttrProxy(object):
    """Translates index lookups into attribute lookups"""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __len__(self):
        return len([x for x in self.module.__dict__['_modules'].keys() if re.match(f'{self.prefix}\d+', x)])

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError
        return getattr(self.module, self.prefix + str(item))

class MultiReadout(nn.Module):
    """A fake list of FFNs for reading out as suggested in
    https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/3 """

    def __init__(self, args: Namespace, atom_targets, bond_targets=None, mol_targets = None,
                 atom_constraints=None, bond_constraints=None):
        """
        :param args: Arguments.
        :param targets: List of targets.
        :param constraints: List indicating which targets are constrained.
        """

        features_size = args.hidden_size
        hidden_size = args.ffn_hidden_size
        num_layers = args.ffn_num_layers
        output_size = args.output_size # is 1
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        if not mol_targets:
            mol_targets = None

        super(MultiReadout, self).__init__()
        i = 0

        for n, a_target in enumerate(atom_targets):
            i += 1
            if atom_constraints[n] == 0:
                constraint = None
            elif (atom_constraints[n] == 1) and (a_target == 'spin_densities'):
                constraint = 'spin_density'
            elif (atom_constraints[n] == 1) and (a_target == 'charges_all_atom'):
                constraint = 'charge'
            self.add_module(f'readout_{n}', FFN(features_size, hidden_size, num_layers,
                                                    output_size, dropout, activation, constraint, ffn_type='atom', use_cuda=args.cuda))

        for j, b_target in enumerate(bond_targets):
            constraint = bond_constraints[j] if bond_constraints and j < len(bond_constraints) else None
            self.add_module(f'readout_{i}', FFN(features_size, hidden_size, num_layers,
                                                output_size, dropout, activation, constraint, ffn_type='bond', use_cuda=args.cuda))
            i += 1

        if mol_targets:
            for mol_target_type in mol_targets:
                if mol_target_type[0] in args.mol_ext_targets:
                    ffn_type_m = 'mol_ext'
                elif mol_target_type[0] in args.mol_int_targets:
                    ffn_type_m = 'mol_int'
                if args.single_mol_tasks:
                    for mol_target in mol_target_type:
                        self.add_module(f'readout_{i}', FFN(features_size, hidden_size, num_layers,
                                                            output_size, dropout, activation, ffn_type=ffn_type_m, use_cuda=args.cuda))
                        i += 1
                else:
                    output_size_m = len(mol_target_type)
                    self.add_module(f'readout_{i}', FFN(features_size, hidden_size, num_layers,
                                                    output_size_m, dropout, activation, ffn_type=ffn_type_m, use_cuda=args.cuda)) 
                    i += 1

        self.ffn_list = AttrProxy(self, 'readout_')

    def forward(self, *input):
        return [ffn(*input) for ffn in self.ffn_list]


class FFN(nn.Module):
    """A Feedforward netowrk reading out properties from fingerprint"""

    def __init__(self, features_size, hidden_size, num_layers, output_size,
                 dropout, activation, constraint=None, ffn_type='atom', attention=False, use_cuda=True):
        """Initializes the FFN.

        args: Arguments.
        constraints: constraints applied to output
        """

        super(FFN, self).__init__()
        if ffn_type == 'atom':
            self.ffn = DenseLayers(features_size, hidden_size,
                                   num_layers, output_size, dropout, activation)
        elif ffn_type == 'bond':
            self.ffn = DenseLayers(2*features_size, hidden_size,
                                   num_layers, output_size, dropout, activation)
        elif ffn_type.startswith('mol'): 
            self.ffn = DenseLayers(features_size, hidden_size,
                                    num_layers, output_size, dropout, activation)
        
        self.ffn_type = ffn_type
        self.attention = attention
        self.use_cuda = use_cuda

        if constraint is not None:
            self.weights_readout = DenseLayers(features_size, hidden_size,
                                               num_layers, output_size, dropout, activation)
            if attention:
                self.weights_readout = DenseLayers(first_linear_dim=hidden_size, output_size=1, num_layers=1,
                                                   dropout=dropout, activation=activation)
            self.constraint = constraint
        else:
            self.constraint = None

    def forward(self, input):
        """
        Runs the FFN on input

        :param input:
        :return:
        """
        a_hidden, a_scope, b_hidden, b_scope, b2br, bond_types, charges, spin_densities = input

        if self.ffn_type == 'atom':
            hidden = a_hidden
            scope = a_scope

            output = self.ffn(hidden)
            if self.attention:
                weights = self.weights_readout(output)

            if self.constraint == 'charge':
                weights = self.weights_readout(hidden)
                constrained_output = []
                for i, (a_start, a_size) in enumerate(scope):
                    constraint = charges[i] # constraint based on actual formal charge and not just 0 for all molecules
                    if a_size == 0:
                        continue
                    else:
                        cur_weights = weights.narrow(0, a_start, a_size)
                        cur_output = output.narrow(0, a_start, a_size)

                        cur_weights_sum = cur_weights.sum()
                        cur_output_sum = cur_output.sum()

                        cur_output = cur_output + cur_weights * \
                                     (constraint - cur_output_sum) / cur_weights_sum
                        constrained_output.append(cur_output)
                output = torch.cat(constrained_output, dim=0)
            #else:
            #    output = output[1:]

            elif self.constraint == 'spin_density':
                weights = self.weights_readout(hidden)
                constrained_output = []
                for i, (a_start, a_size) in enumerate(scope):
                    constraint = spin_densities[i] # constraint based on Num Radicals electrons
                    if a_size == 0:
                        continue
                    else:
                        cur_weights = weights.narrow(0, a_start, a_size)
                        cur_output = output.narrow(0, a_start, a_size)
                        
                        cur_weights_sum = cur_weights.sum()
                        cur_output_sum = cur_output.sum()

                        cur_output = cur_output + cur_weights * \
                                     (constraint - cur_output_sum) / cur_weights_sum
                        constrained_output.append(cur_output)
                output = torch.cat(constrained_output, dim=0)
            else:
                output = output[1:]
                
        elif self.ffn_type == 'bond':

            forward_bond = b_hidden[b2br[:, 0]]
            backward_bond = b_hidden[b2br[:, 1]]

            b_hidden = torch.cat([forward_bond, backward_bond], dim=1)

            output = self.ffn(b_hidden) + bond_types.reshape(-1, 1)

        elif self.ffn_type.startswith('mol'):
            scope = a_scope
            _,b = a_hidden.shape
            hidden = torch.empty((len(a_scope),b), dtype=torch.float64)
            for i, (a_start, a_size) in enumerate(scope):
                mol_a_hidden = a_hidden[a_start:a_start+a_size]
                if self.ffn_type == 'mol_ext':
                    mol_hidden = torch.sum(mol_a_hidden, 0)
                elif self.ffn_type == 'mol_int':
                    mol_hidden = torch.mean(mol_a_hidden, 0)
                hidden[i] = mol_hidden
            
            if self.use_cuda:
                hidden = hidden.cuda()
            output = self.ffn(hidden.float())

        return output


class DenseLayers(nn.Module):
    "Dense layers"

    def __init__(self,
                 first_linear_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 dropout: nn.Module,
                 activation) -> nn.Sequential:
        """
        :param first_linear_dim:
        :param hidden_size:
        :param num_layers:
        :param output_size:
        :param dropout:
        :param activation:
        """
        super(DenseLayers, self).__init__()
        if num_layers == 1:
            layers = [
                dropout,
                nn.Linear(first_linear_dim, output_size)
            ]
        else:
            layers = [
                dropout,
                nn.Linear(first_linear_dim, hidden_size)
            ]
            for _ in range(num_layers - 2):
                layers.extend([
                    activation,
                    dropout,
                    nn.Linear(hidden_size, hidden_size),
                ])
            layers.extend([
                activation,
                dropout,
                nn.Linear(hidden_size, output_size),
            ])

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.dense_layers(input)
