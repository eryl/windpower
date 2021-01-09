import io
import pickle
from typing import Sequence, Mapping, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
import numpy as np

from windpower.dataset import SiteDataset
from windpower.mltrain.train import MinibatchModel
from windpower.mltrain.performance import LowerIsBetterMetric, EvaluationMetric


class DummyModel(object):
    def fit(self, sites, history, forecast, target):
        return 1

    def predict(self, sites, history, forecast, target):
        return 1


class FeedforwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p=0.5, activation_fn=None):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation_fn = activation_fn() if activation_fn is not None else None

    def forward(self, x):
        #x = self.bn(input)
        x = self.dropout(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


@dataclass
class TorchWrapperConfig:
    batch_size: int
    device: torch.device
    model_class: Type
    model_args: Sequence
    model_kwargs: Mapping


class TorchWrapper(MinibatchModel):
    def __init__(self, config: TorchWrapperConfig,
                 training_dataset: SiteDataset = None,
                 validation_dataset: SiteDataset = None,
                 test_dataset: SiteDataset=None,
                 rng=None):
        self.config = config

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        # We need to figure out the input and output dimensions
        dataset = training_dataset if training_dataset is not None else validation_dataset if validation_dataset is not None else test_dataset
        if dataset is not None:
            observation = dataset[0]
            x = observation['x']
            y = observation['y']
            input_dim = x.shape[0]
            if len(y.shape) == 0:
                output_dim = 1
            else:
                output_dim = y.shape[0]

            input_dim = input_dim
            output_dim = output_dim
        else:
            raise ValueError("No dataset given, can't infer input and output dimensions")

        self.device = config.device
        self.model = self.config.model_class(*self.config.model_args, input_dim = input_dim, output_dim = output_dim, **self.config.model_kwargs, rng=rng)
        self.model = self.model.to(self.device)

    def prepare_dataset(self, dataset, shuffle=True):
        def collate_fn(batch):
            x = np.stack([row['x'] for row in batch], axis=0)
            y = np.stack([row['y'] for row in batch], axis=0)
            time = [row['target_time']for row in batch]
            var_info = [row['variable_info'] for row in batch]
            return dict(x=torch.tensor(x, dtype=torch.float32), y=torch.tensor(y, dtype=torch.float32), target_time=time, variable_info=var_info)

        return DataLoader(dataset, batch_size=self.config.batch_size,
                          shuffle=shuffle, drop_last=False, collate_fn=collate_fn)

    def get_metadata(self):
        return dict()

    def fit_batch(self, batch) -> Dict:
        x = batch['x']
        y = batch['y']
        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self.model.train()
        self.model.optim.zero_grad()

        y_hat = self.model(x)
        loss = self.model.loss_fn(y_hat.squeeze(), y)
        loss.backward()
        self.model.optim.step()
        return {'training_loss': loss.detach().item()}

    def evaluate_batch(self, batch) -> Dict:
        self.model.eval()
        with torch.no_grad():
            x = batch['x']
            y = batch['y']
            x = x.to(device=self.device)
            y = y.to(device=self.device)

            y_hat = self.model(x)
            loss = self.model.loss_fn(y_hat.squeeze(), y)
            mean_absolute_error = torch.abs(y_hat.clip(0,1) - y).mean()
            return {'mean_absolute_error': mean_absolute_error.item(), 'evaluation_loss': loss.detach().item()}

    def evaluation_metrics(self) -> List[EvaluationMetric]:
        return [LowerIsBetterMetric('mean_absolute_error'), LowerIsBetterMetric('evaluation_loss')]

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            y_hat = self.model(x).clip(0, 1)
            return y_hat.cpu().numpy()

    def save(self, save_path: Path) -> Union[str, Path]:
        save_path = save_path.with_name(save_path.name + '.pth')  # The filename might contain dots in floating point valuies, we shouldnt use with_suffix
        #self.to('cpu')
        torch.save(self.model.to('cpu').state_dict(), save_path)
        self.model.to(self.device)
        return save_path

    def load(self, load_path: Path):
        state_dict = torch.load(load_path)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        return self


@dataclass
class NNTabularConfig:
    n_layers: int
    layer_size: int
    dropout_p: float
    optim_class: type
    skip_connections: bool = True
    limit_range: bool = True
    optim_args: Sequence = field(default_factory=set)
    optim_kwargs: Mapping = field(default_factory=dict)


class NNTabularModel(nn.Module):
    def __init__(self, config: NNTabularConfig, *, input_dim, output_dim, rng=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.config = config
        self.layers = None

        self.n_layers = self.config.n_layers
        self.layer_size = self.config.layer_size

        layers = []
        self.skip_connections = self.config.skip_connections
        self.limit_range = self.config.limit_range

        self.projection_layer = nn.Linear(self.input_dim, self.config.layer_size,
                                          bias=False)  # the projection layer makes sure the input is of the same dimension as the other layers so we can use residual connections
        dim_from_below = self.config.layer_size
        for i in range(self.config.n_layers):
            layers.append(FeedforwardBlock(dim_from_below, self.config.layer_size, activation_fn=nn.ReLU))
            dim_from_below = self.config.layer_size
        self.output_layer = FeedforwardBlock(dim_from_below, self.output_dim, activation_fn=None)
        self.layers = nn.ModuleList(layers)
        self.loss_fn = nn.MSELoss()
        self.optim = self.config.optim_class(self.parameters(), *self.config.optim_args,
                                             **self.config.optim_kwargs)  # lr=learning_rate, weight_decay=adam_wd, betas=(adam_beta1, adam_beta2), eps=adam_eps, )


    def forward(self, x):
        x = self.projection_layer(x)  # Make sure x is of the right dimension for the residual connections to work
        for block in self.layers:
            if self.skip_connections:
                x = block(x) + x
            else:
                x = block(x)
        x = self.output_layer(x)
        if self.limit_range:
            x = torch.sigmoid(x)*1.1 - 0.05
        return x


class CNNBaseLine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        pass


class LSTMSiteEmbedding(nn.Module):
    def __init__(self, *, site_ids, input_size, rnn_size, rnn_layers, embedding_dim):
        super().__init__()
        self.site_ids = site_ids
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.site_embeddings = nn.Embedding(len(site_ids), embedding_dim=embedding_dim)
        self.history_encoder = nn.LSTM(input_size, rnn_size, rnn_layers, batch_first=True)
        self.forecast_rnn = nn.LSTM(input_size+rnn_size*rnn_layers, rnn_size, rnn_layers, batch_first=True)
        self.forecast_function = nn.Sequential()

    def fit(self, sites, history, forecast, target):
        representation = self(sites, history, forecast, target)

    def predict(self, sites, history, forecast, target):
        representation = self(sites, history, forecast, target)

    def forward(self, sites, history, forecast, target):
        batch_size = forecast.shape[0]
        sequence_length = forecast.shape[1]
        site_representations = self.site_embeddings[sites]
        hh_0, hc_0 = (torch.zeros((self.rnn_layers, batch_size, self.rnn_size), device=history.device),
                      torch.zeros((self.rnn_layers, batch_size, self.rnn_size), device=history.device))
        # We only care about the last value of the output
        output, (hh, hc) = self.history_encoder(history, (hh_0, hc_0))
        # Take the last hidden state of all the layers of all the batches
        history_encoding = hh[..., -1]  # The history encoding has shape (num_layers, batch_size, hidden_size), we
                                        # combine the outputs of all layers to a single vector as its history representation
        history_encoding_reshaped = history_encoding.transpose(0,1).view(batch_size, 1, self.rnn_layers*self.rnn_size)  #put batch first
        # We want to add the history encoding to each time step. The shape right now is (batch_size, 1, rnn_layers*rnn_size),
        # we want to repeat the matrix along the sequence dimension, getting the shape (batch_size, sequence_length, rnn_layers*rnn_size)
        history_encoding_expanded = history_encoding_reshaped.expand(batch_size, sequence_length, self.rnn_layers*self.rnn_size)
        fh_0, fc_0 = (torch.zeros((self.rnn_layers, batch_size, self.rnn_size), device=history.device),
                      torch.zeros((self.rnn_layers, batch_size, self.rnn_size), device=history.device))
        forecast_input = torch.cat([history_encoding_expanded, forecast], dim=-1)
        forecast_representation = self.forecast_rnn(forecast_input, (fh_0, fc_0))
        return forecast_representation


class CNNSiteEmbedding(nn.Module):
    def __init__(self, site_ids, embedding_dim):
        super().__init__()
        self.site_ids = site_ids
        self.site_embeddings = nn.Embedding(len(site_ids), embedding_dim=embedding_dim)


    def fit(self, sites, input, target):

        ...

    def predict(self, sites, input, target):
        ...

    def forward(self, sites, input, target):
        site_representations = self.site_embeddings[sites]



class CNNAttentionDynamicSiteEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        pass


