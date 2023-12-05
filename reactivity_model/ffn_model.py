import torch
import copy
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class ForwardFFN(pl.LightningModule):

    def __init__(self,
                 hidden_size: int,
                 layers: int = 2,
                 dropout: float = 0.0,
                 learning_rate: float = 1e-3,
                 input_dim: int = 10,
                 output_dim: int = 1,
                 lr_ratio: float = 0.97,
                 **kwargs):
        """
        Feed Forward Neural Network model

        Args:
            hidden_size (int): hidden_size
            layers (int): layers
            dropout (float): dropout
            learning_rate (float): learning_rate
            min_lr (float): min_lr
            input_dim (int): input_dim
            output_dim (int): output_dim
            kwargs:
        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.learning_rate = learning_rate
        self.lr_ratio = lr_ratio
        self.loss_fn = self.mse_loss

        self.all_layers = nn.Sequential(
            MLPBlocks(input_size=self.input_dim,
                      hidden_size=self.hidden_size,
                      dropout=self.dropout,
                      num_layers=self.layers),
            nn.Sequential(
                nn.Linear(self.hidden_size,
                          self.output_dim),
                nn.Identity())
        )

    def forward(self, desc):
        pred = self.all_layers(desc).flatten()
        return pred

    def mse_loss(self, pred, targ, **kwargs):
        """ mse_loss.

        Args:
            pred (torch.tensor): Predictions
            targ (torch.tensor): Targets
        """
        mse_loss = F.mse_loss(pred, targ)
        return {"loss": mse_loss}

    def training_step(self, batch, batch_idx):
        """training_step.

        Args:
            batch:
            batch_idx:
        """
        x, y = batch
        preds = self.forward(x)
        loss_dict = self.loss_fn(preds, y)
        self.log("train_loss", loss_dict.get("loss"))
        return loss_dict

    def validation_step(self, batch, batch_idx):
        """validation_step.

        Args:
            batch:
            batch_idx:
        """
        x, y = batch
        preds = self.forward(x)
        loss_dict = self.loss_fn(preds, y)
        self.log("val_loss", loss_dict.get("loss"))
        return loss_dict

    def test_step(self, batch, batch_idx):
        """test_step.

        Args:
            batch:
            batch_idx:
        """
        x, y = batch
        preds = self.forward(x)
        loss_dict = self.loss_fn(preds, y)
        self.log("test_loss", loss_dict.get("loss"))
        return loss_dict

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """test_step.

                Args:
                    batch:
                    batch_idx:
                    dataloader_idx:
                """
        x, y = batch
        preds = self.forward(x)
        return preds

    def configure_optimizers(self):
        """configure_optimizers.
        """

        non_frozen_parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(non_frozen_parameters,
                                     lr=self.learning_rate,
                                     weight_decay=0.0)

        start_lr = self.learning_rate
        lr_ratio = self.lr_ratio
        lr_lambda = lambda epoch: start_lr * lr_ratio**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lr_lambda)
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "epoch"
            }
        }
        return ret


class MLPBlocks(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        num_layers: int,
    ):
        """
        Initial and hidden layers

        Args:
            input_size (int): input_size
            hidden_size (int): hidden_size
            dropout (float): dropout
            num_layers (int): num_layers
        """
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(input_size, hidden_size)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = get_clones(middle_layer, num_layers)

    def forward(self, x):
        """forward.

        Args:
            x:
        """
        output = x
        output = self.input_layer(x)
        output = self.activation(output)
        for layer_index, layer in enumerate(self.layers):
            output = layer(output)
            output = self.dropout_layer(output)
            output = self.activation(output)
        return output


def get_clones(module, N):
    """get_clones.

    Args:
        module:
        N:
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
