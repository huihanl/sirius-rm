"""
Contains an implementation of RSSMs.
"""

import textwrap
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.models.base_nets import Module, MLP
from robomimic.models.obs_nets import MIMO_MLP

import wandb

from . import rssm_modules as rnn

from torch import Tensor
from typing import Any, Optional, Tuple, Callable, Dict, List, Optional, TypeVar, Union
import torch.distributions as D

def init_weights_gru(m):
    if type(m) == nn.GRUCell or type(m) == rnn.GRUCell:
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)
    if type(m) == rnn.NormGRUCell or type(m) == rnn.NormGRUCellLateReset:
        nn.init.xavier_uniform_(m.weight_ih.weight.data)
        nn.init.orthogonal_(m.weight_hh.weight.data)


class RSSMCore(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, hidden_dim, gru_layers, gru_type, device, use_network_za=True, diff_za_dim=False, prior_larger=False):
        super().__init__()
        self.cell = RSSMCell(embed_dim, action_dim, deter_dim, stoch_dim, hidden_dim, gru_layers, gru_type, 
                             use_network_za=use_network_za,
                             diff_za_dim=diff_za_dim,
                             larger=prior_larger)
        self.device = device
        for m in self.modules():
            init_weights_gru(m)

    def forward(self,
                embeds,              # tensor(T, B, E)
                actions,             # tensor(T, B, A)
                h_t,                 # tensor(B, D)
                z_t,                 # tensor(B, S)
                ):

        T, B = embeds.shape[:2]

        # posterior
        posts = []
        states_h = []
        post_z_samples = []

        for i in range(T):

            # confirm initial state is 0
            #if T > 1 and i == 0:
            #    assert torch.all(h_t == 0) and torch.all(z_t == 0) # and torch.all(actions[i] == 0)
            
            post_z, h_t, z_t = self.cell.forward(embeds[i], actions[i], h_t, z_t)  # iteratively pass in h_t, z_t
            posts.append(post_z)
            states_h.append(h_t)
            post_z_samples.append(z_t)

        posts = torch.stack(posts)          # (T,B,2S)
        states_h = torch.stack(states_h)    # (T,B,D)
        post_z_samples = torch.stack(post_z_samples)      # (T,B,S)

        # priors
        priors = self.cell.batch_prior(states_h)  # (T,B,2S)
        features = self.to_feature(states_h, post_z_samples)   # (T,B,D+S)

        states = (states_h, post_z_samples)

        return (
            priors,                      # tensor(T,B,2S)
            posts,                       # tensor(T,B,2S)
            post_z_samples,              # tensor(T,B,S)
            features,                    # tensor(T,B,D+S)
            states,
            (h_t.detach(), z_t.detach()),
        )

    def forward_zero_actions(self,
                             embeds,  # tensor(T, B, E)
                             actions,  # tensor(T, B, A)
                             initial_h_t,  # tensor(B, D) # zero init
                             initial_z_t,  # tensor(B, S) # zero init
                             ):

        T, B = embeds.shape[:2]

        # posterior
        posts = []
        states_h = []
        post_z_samples = []

        assert torch.all(initial_h_t == 0) and torch.all(initial_z_t == 0) 
        assert self._zero_actions(actions)

        for i in range(T):

            post_z, h_t, z_t = self.cell.forward(embeds[i],
                                                 actions[i],
                                                 initial_h_t,
                                                 initial_z_t)  # always the initial h_t, z_t
            posts.append(post_z)
            states_h.append(h_t)
            post_z_samples.append(z_t)

        posts = torch.stack(posts)  # (T,B,2S)
        states_h = torch.stack(states_h)  # (T,B,D)
        post_z_samples = torch.stack(post_z_samples)  # (T,B,S)

        # priors
        # priors = self.cell.batch_prior(states_h)  # (T,B,2S)
        features = self.to_feature(states_h, post_z_samples)  # (T,B,D+S)

        states = (states_h, post_z_samples)

        return (
            None,  # tensor(T,B,2S)
            posts,  # tensor(T,B,2S)
            post_z_samples,  # tensor(T,B,S)
            features,  # tensor(T,B,D+S)
            states,
            (h_t.detach(), z_t.detach()),
        )

    def to_feature(self, h: Tensor, z: Tensor) -> Tensor:
        return torch.cat((h, z), -1)

    def zdistr(self, pp: Tensor) -> D.Distribution:
        return self.cell.zdistr(pp)

    def init_state(self, batch_size):
        return self.cell.init_state(batch_size)

    def _zero_actions(self, actions):
        # currently, zero action is the default spacemouse action (gripper action = -1)
        return torch.all(actions[:, :, :6] == 0) and torch.all(actions[:, :, -1] == -1)

class RSSMCell(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, hidden_dim, gru_layers, gru_type, use_network_za, diff_za_dim=False, larger=False):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        self.deter_next = DeterNext(stoch_dim,
                                    deter_dim,
                                    action_dim,
                                    hidden_dim,
                                    gru_layers,
                                    gru_type,
                                    use_network_za=use_network_za,
                                    diff_za_dim=diff_za_dim,
                                    )

        self.posterior_next = PosteriorNext(
                                    embed_dim,
                                    stoch_dim,
                                    deter_dim,
                                    hidden_dim,
                                    )

        if larger:
            self.prior_next = PriorNextLarge(
                                    stoch_dim,
                                    deter_dim,
                                    hidden_dim,
                                    )
        else:
            self.prior_next = PriorNext(
                                        stoch_dim,
                                        deter_dim,
                                        hidden_dim,
                                        )

    def forward(self,
                embed,                 # tensor(B,E)
                a_t,                   # tensor(B,A)
                h_t,
                z_t,
                ):

        B = a_t.shape[0]

        # get deterministic h_t+1
        h_t_next = self.deter_next(z_t, a_t, h_t)

        # get stochastic posterior z_t+1
        post_z = self.posterior_next(embed, h_t_next)

        # sample posterior z_t+1
        post_z_distr = self.zdistr(post_z)
        post_z_sample = post_z_distr.rsample().reshape(B, -1)

        return (
            post_z,                         # tensor(B, 2*S)
            h_t_next,
            post_z_sample,
        )

    def batch_prior(self,
                    h_t_next,     # tensor(T, B, D)
                    ):
        prior = self.prior_next(h_t_next)
        return prior # tensor(T, B, 2S)

    def zdistr(self, z) -> D.Distribution:
        return self.diag_normal(z)

    def diag_normal(self, z, min_std=0.1, max_std=2.0):
        # TODO: Why?
        mean, std = z.chunk(2, -1)
        std = max_std * torch.sigmoid(std) + min_std
        return D.independent.Independent(D.normal.Normal(mean, std), 1)

    def init_state(self, batch_size):
        device = next(self.deter_next.gru.parameters()).device
        return (
            torch.zeros((batch_size, self.deter_dim), device=device), # h_t
            torch.zeros((batch_size, self.stoch_dim), device=device), # z_t
        )

class DeterNext(nn.Module):
    """
    Input: z_t (stoch), a_t (action), h_t (deter)
    Output: h_t+1 (deter)
    """
    def __init__(
        self,
        stoch_dim,
        deter_dim,
        action_dim,
        hidden_dim,
        gru_layers,
        gru_type,
        layer_func=nn.Linear,
        layer_func_kwargs=None,
        activation=nn.ReLU,
        layer_norm=nn.LayerNorm,
        dropouts=None,
        normalization=False,
        output_activation=None,
        use_network_za=True,
        diff_za_dim=False,
    ):
        super(DeterNext, self).__init__()
        if use_network_za:
            if diff_za_dim:
                assert hidden_dim % 4 == 0
                self.z_mlp = nn.Linear(stoch_dim, hidden_dim // 4 * 3)
                self.a_mlp = nn.Linear(action_dim, hidden_dim // 4)
            else:
                self.z_mlp = nn.Linear(stoch_dim, hidden_dim // 2)
                self.a_mlp = nn.Linear(action_dim, hidden_dim // 2, bias=False)
            self.gru = rnn.GRUCellStack(hidden_dim, deter_dim, gru_layers, gru_type)
            self._layer_norm = layer_norm(hidden_dim, eps=1e-3)
        else:
            self.gru = rnn.GRUCellStack(stoch_dim + action_dim, deter_dim, gru_layers, gru_type)
            self._layer_norm = layer_norm(stoch_dim + action_dim, eps=1e-3)

        self.use_network_za = use_network_za
        self._hidden_dim = hidden_dim
        self._output_dim = deter_dim

    def output_shape(self, input_shape=None):
        return [self._output_dim]

    def forward(self, z_t, a_t, h_t):
        if self.use_network_za:
            x = torch.cat((self.z_mlp(z_t), self.a_mlp(a_t)), dim=1)
        else:
            x = torch.cat((z_t, a_t), dim=1)

        x = self._layer_norm(x)
        za = F.elu(x)
        h_t_next = self.gru(za, h_t) 
        return h_t_next

class PosteriorNext(nn.Module):
    """
    Input: h_t+1 (deter), obs embed (embed)
    Output: posterior z_t+1 (stoch)
    """
    def __init__(
            self,
            embed_dim,
            stoch_dim,
            deter_dim,
            hidden_dim,
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            layer_norm=nn.LayerNorm,
            dropouts=None,
            normalization=False,
            output_activation=None,
            middle_layer=False,
    ):
        super(PosteriorNext, self).__init__()

        self.post_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.post_mlp_embed = nn.Linear(embed_dim, hidden_dim)
        
        self.middle_layer = middle_layer
        if middle_layer:
            self.middle_mlp = nn.Linear(hidden_dim, hidden_dim)
        
        self.post_mlp = nn.Linear(hidden_dim, stoch_dim * 2)

        self._hidden_dim = hidden_dim
        self._output_dim = deter_dim
        self._layer_norm = layer_norm(hidden_dim, eps=1e-3)
        self._output_dim = stoch_dim * 2 # mean, std for sampling

    def output_shape(self, input_shape=None):
        return [self._output_dim]

    def forward(self, embed, h_t_next):
        x = self.post_mlp_h(h_t_next) + self.post_mlp_embed(embed)
        x = self._layer_norm(x)
        post_in = F.elu(x)

        if self.middle_layer:
            post_in = self.middle_mlp(post_in)
            post_in = self._layer_norm(post_in)
            post_in = F.elu(post_in)

        post = self.post_mlp(post_in)
        return post

class PriorNext(nn.Module):
    """
    Input: h_t+1 (deter)
    Output: prior z_t+1 (stoch)
    """
    def __init__(
            self,
            stoch_dim,
            deter_dim,
            hidden_dim,
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            layer_norm=nn.LayerNorm,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        super(PriorNext, self).__init__()

        self.prior_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.prior_mlp = nn.Linear(hidden_dim, stoch_dim * 2)

        self._hidden_dim = hidden_dim
        self._layer_norm = layer_norm(hidden_dim, eps=1e-3)
        self._output_dim = stoch_dim * 2 # mean, std for sampling

    def output_shape(self, input_shape=None):
        return [self._output_dim]

    def forward(self, h_t_next):
        x = self.prior_mlp_h(h_t_next)
        x = self._layer_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)  # tensor(B,2S)
        return prior

class PriorNextLarge(nn.Module):
    """
    Input: h_t+1 (deter)
    Output: prior z_t+1 (stoch)
    """
    def __init__(
            self,
            stoch_dim,
            deter_dim,
            hidden_dim,
            num_layers=2,
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=F.elu,
            layer_norm=nn.LayerNorm,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        super(PriorNextLarge, self).__init__()

        self._output_dim = stoch_dim * 2 # mean, std for sampling

        layer_dims = [hidden_dim] * num_layers
        activation=nn.ELU

        self._mlp = MLP(
                input_dim=deter_dim,
                output_dim=stoch_dim * 2,
                layer_dims=layer_dims,
                layer_func=layer_func,
                activation=activation,
                output_activation=None,
                normalization=True,
            )
        
        print("PriorNextLarge:")
        print(self._mlp)

    def output_shape(self, input_shape=None):
        return [self._output_dim]

    def forward(self, h_t_next):
        prior = self._mlp(h_t_next)  # tensor(B,2S)
        return prior
