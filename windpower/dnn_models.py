import io
import pickle

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
import numpy as np

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

    def forward(self, input):
        x = self.bn(input)
        x = self.dropout(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x

class NNTabularModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, layer_size, device=None, dropout_p=0.5, rng_seed=1729,
                 optim_args=None, optim_kwargs=None, skip_connections=False, limit_range=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.init_args = (input_dim, output_dim, n_layers, layer_size)
        self.init_kwargs = dict(dropout_p=dropout_p, rng_seed=rng_seed,
                                skip_connections=skip_connections,
                                optim_args=optim_args,
                                optim_kwargs=optim_kwargs,
                                limit_range=limit_range)
        if optim_args is None:
            optim_args = set()
        if optim_kwargs is None:
            optim_kwargs = dict()
        self.rng = np.random.RandomState(rng_seed)

        layers = []
        self.skip_connections = skip_connections
        self.limit_range = limit_range
        self.input_transform = nn.Linear(input_dim, layer_size)
        dim_from_below = layer_size
        for i in range(n_layers):
            layers.append(FeedforwardBlock(dim_from_below, layer_size, activation_fn=nn.ReLU))
            dim_from_below = layer_size
        self.output_layer = FeedforwardBlock(dim_from_below, output_dim, activation_fn=None)
        self.layers = nn.ModuleList(layers)
        self.loss_fn = nn.MSELoss()
        self.optim = AdamW(self.parameters(), *optim_args, **optim_kwargs) # lr=learning_rate, weight_decay=adam_wd, betas=(adam_beta1, adam_beta2), eps=adam_eps, )
        if device is not None:
            self.to(device)
            for l in self.layers:
                l.to(device)
        self.device = device

    def get_metadata(self):
        metadata = dict(input_dim=self.input_dim,
                        output_dim=self.output_dim,
                        n_layers=self.n_layers,
                        layer_size=self.layer_size)
        metadata.update(self.init_kwargs)
        return metadata

    def forward(self, input):
        x = self.input_transform(input)
        for block in self.layers:
            if self.skip_connections:
                x = block(x) + x
            else:
                x = block(x)
        x = self.output_layer(x)
        if self.limit_range:
            x = torch.sigmoid(x)*1.1 - 0.05
        return x

    def fit(self, batch):
        self.train(True)
        self.optim.zero_grad()
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self(x)
        loss = self.loss_fn(y_hat.squeeze(), y)
        loss.backward()
        self.optim.step()
        return {'training_loss': loss.detach().item()}

    def evaluate(self, batch):
        self.train(False)
        with torch.no_grad():
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self(x)
            loss = self.loss_fn(y_hat.squeeze(), y)
            return {'evaluation_loss': loss.detach().item()}

    def save(self, path):
        torch_model_state = io.BytesIO()
        torch.save(self.state_dict(), torch_model_state)
        save_state = dict(init_args=self.init_args,
                          init_kwargs=self.init_kwargs,
                          model_state=torch_model_state.getvalue())
        with open(path, 'wb') as fp:
            pickle.dump(save_state, fp)

    @classmethod
    def load(cls, path, device=None):
        with open(path, 'rb') as fp:
            saved_state = pickle.load(fp)
        torch_model_state = torch.load(io.BytesIO(saved_state['model_state']))
        if 'init_args' in saved_state:
            init_args = saved_state['init_args']
        else:
            init_args = tuple()
        if 'init_kwargs' in saved_state:
            init_kwargs = saved_state['init_kwargs']
        else:
            init_kwargs = dict()
        model = cls(*init_args, **init_kwargs)
        model.load_state_dict(torch_model_state)
        if device is not None:
            model.to(device)
        return model



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


