from argparse import Namespace

import torch
import torch.nn as nn

from .mpn import MPN
from .ffn import MultiReadout
from utilities.utils_nn import initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self):
        """
        Initializes the MoleculeModel.
        """
        super(MoleculeModel, self).__init__()


    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        # Create readout layers (several FFN)
        self.readout = MultiReadout(args, args.atom_targets, args.bond_targets, args.mol_targets,
                                    args.atom_constraints, args.bond_constraints)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        output_all = self.readout(self.encoder(*input))

        return output_all

def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    args.output_size = 1

    model = MoleculeModel()
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
