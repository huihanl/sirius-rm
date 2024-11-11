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
from robomimic.models.base_nets import Module, MLP, ResidualMLP
from robomimic.models.obs_nets import MIMO_MLP
from robomimic.models.vae_nets import VAE
import robomimic.models.vae_nets as VAENets
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import res_mlp_args_from_config_vae


# Unused
class Dynamics(nn.Module):

    def __init__(self, embed_dim, action_dim, hidden_dim, action_network_dim=0, num_layers=2, use_res_mlp=False):
        super().__init__()
        self.cell = DynamicsCell(embed_dim, 
                                 action_dim, 
                                 hidden_dim, 
                                 action_network_dim=action_network_dim, 
                                 num_layers=num_layers,
                                 use_res_mlp=use_res_mlp)

    def forward(self,
                embeds,              # tensor(T, B, E)
                actions,             # tensor(T, B, A)
                ):

        T, B = embeds.shape[:2]

        pred_embeds = []
        embed = embeds[0]
        for i in range(T-1):
            embed = self.cell.forward(embed, actions[i])
            pred_embeds.append(embed)
        pred_embeds = torch.stack(pred_embeds) 

        return pred_embeds

# Unused
class DynamicsCell(nn.Module):
    def __init__(self, embed_dim, action_dim, hidden_dim, action_network_dim=0, num_layers=2, use_res_mlp=False):
        super().__init__()

        self.action_network_dim = action_network_dim
        if action_network_dim > 0:
            self._action_network = MLP(
                input_dim=action_dim,
                output_dim=action_network_dim,
                layer_dims=[hidden_dim],
                activation=nn.ELU,
                output_activation=None,
                normalization=True,
            )
            input_dim = embed_dim + action_network_dim
        else:
            input_dim = embed_dim + action_dim
        
        if use_res_mlp:
            self._mlp = nn.Sequential(
                ResidualMLP(
                    input_dim=input_dim,
                    activation=nn.ReLU,
                    output_activation=None,
                    num_blocks=4,
                    hidden_dim=hidden_dim,
                    normalization=True
                ),
                nn.Linear(hidden_dim, embed_dim)
            )

        else:
            self._mlp = MLP(
                input_dim=input_dim,
                output_dim=embed_dim,
                layer_dims=[hidden_dim] * num_layers,
                activation=nn.ELU,
                output_activation=None,
                normalization=True,
            )

    def forward(self, embed, action):
        if self.action_network_dim > 0:
            action = self._action_network(action)
        pred = torch.cat([embed, action], dim=-1)
        pred = self._mlp(pred)

        return pred 


########


class DynamicsVAE(nn.Module):

    def __init__(self, embed_dim, action_dim, device, algo_config, obs_config, use_history=False, use_real=True):
        super().__init__()

        self.algo_config = algo_config
        self.obs_config = obs_config

        self.use_history = use_history
        self.use_real = use_real

        self.cell = DynamicsVAE_Cell(
                        embed_dim=embed_dim,
                        action_dim=action_dim,
                        device=device,
                        use_history=use_history,
                        encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(obs_config.encoder),
                        **VAENets.vae_args_from_config(algo_config.vae),
                        enc_use_res_mlp=algo_config.vae.enc_use_res_mlp,
                        dec_use_res_mlp=algo_config.vae.dec_use_res_mlp,
                        **res_mlp_args_from_config_vae(algo_config.res_mlp),  
                    )

    def forward(self,
                embeds,              # tensor(T,      B, E)
                actions,             # tensor(T // 2, B, A)
                ):

        T, B = embeds.shape[:2]

        kl_loss = []
        reconstruction_loss = []
        pred_embeds = []

        embed_first_half = embeds[:T//2]
        embed_second_half = embeds[T//2:]

        curr_history = embeds[1:T//2+1]

        embed = embed_second_half[0]

        assert actions.shape[0] == embed_second_half.shape[0] == T // 2

        for i in range(T//2 - 1):

            assert len(pred_embeds) == i

            if self.use_history:
                if self.use_real:
                    curr_embed = torch.cat([embed_first_half[i+1:], embed_second_half[:i+1]], dim=0)
                else:
                    if i == 0:
                        curr_embed = curr_history[i:]
                    else:
                        curr_embed = torch.cat([curr_history[i:], torch.stack(pred_embeds[:i])], dim=0)
                
                assert curr_embed.shape[0] == T // 2 # (T, B, E)

                curr_embed = torch.permute(curr_embed, (1, 0, 2)) # (B, T, E)
                curr_embed = curr_embed.reshape(-1, curr_embed.shape[1] * curr_embed.shape[2]) # (B, T * E)

                assert curr_embed.shape[0] == B

            else:
                if self.use_real:
                    curr_embed = embed_second_half[i]
                else:
                    curr_embed = embed
            
            action = actions[i]
            next_embed=embed_second_half[i+1]

            vae_outputs = self.cell.forward(
                next_embed=next_embed, 
                curr_embed=curr_embed, 
                action=action,
            )
            embed = vae_outputs["reconstructions"]['next_embed']
            kl_loss.append(vae_outputs["kl_loss"])
            reconstruction_loss.append(vae_outputs["reconstruction_loss"])
            pred_embeds.append(embed)
        
        pred_embeds = torch.stack(pred_embeds)
        kl_loss = torch.stack(kl_loss) 
        reconstruction_loss = torch.stack(reconstruction_loss)
        total_loss = reconstruction_loss + self.algo_config.vae.kl_weight * kl_loss

        return OrderedDict(
            recons_loss=reconstruction_loss,
            kl_loss=kl_loss,
            dyn_loss=total_loss,
            pred_obs_embedding=pred_embeds,
        )


class DynamicsVAE_Cell(Module):
    """
    Modelling dynamics with a VAE.
    """
    def __init__(
        self,
        embed_dim,
        action_dim,
        encoder_layer_dims,
        decoder_layer_dims,
        latent_dim,
        device,
        use_history=True,
        conditioned_on_obs=True,
        decoder_is_conditioned=True,
        decoder_reconstruction_sum_across_elements=False,
        latent_clip=None,
        prior_learn=False,
        prior_is_conditioned=False,
        prior_layer_dims=(),
        prior_use_gmm=False,
        prior_gmm_num_modes=10,
        prior_gmm_learn_weights=False,
        prior_use_categorical=False,
        prior_categorical_dim=10,
        prior_categorical_gumbel_softmax_hard=False,
        goal_shapes=None,
        encoder_kwargs=None,
        enc_use_res_mlp=False,
        dec_use_res_mlp=False,
        res_mlp_kwargs=None,
        action_network=True,
    ):
        super(DynamicsVAE_Cell, self).__init__()

        self.action_network = action_network

        # input: next_obs embedding
        self.input_shapes = OrderedDict()
        self.input_shapes["next_embed"] = (embed_dim,)

        # condition: curr_obs embedding, action
        self.condition_shapes = OrderedDict()
        if use_history:
            self.condition_shapes["curr_embed"] = (embed_dim * 10,)
        else:
            self.condition_shapes["curr_embed"] = (embed_dim,)

        if self.action_network:
            self.condition_shapes["action"] = (embed_dim,)
        else:
            self.condition_shapes["action"] = (action_dim,)

        self._vae = VAE(
            input_shapes=self.input_shapes,
            output_shapes=self.input_shapes,
            encoder_layer_dims=encoder_layer_dims,
            decoder_layer_dims=decoder_layer_dims,
            latent_dim=latent_dim,
            device=device,
            condition_shapes=self.condition_shapes,
            decoder_is_conditioned=decoder_is_conditioned,
            decoder_reconstruction_sum_across_elements=decoder_reconstruction_sum_across_elements,
            latent_clip=latent_clip,
            prior_learn=prior_learn,
            prior_is_conditioned=prior_is_conditioned,
            prior_layer_dims=prior_layer_dims,
            prior_use_gmm=prior_use_gmm,
            prior_gmm_num_modes=prior_gmm_num_modes,
            prior_gmm_learn_weights=prior_gmm_learn_weights,
            prior_use_categorical=prior_use_categorical,
            prior_categorical_dim=prior_categorical_dim,
            prior_categorical_gumbel_softmax_hard=prior_categorical_gumbel_softmax_hard,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
            enc_use_res_mlp=enc_use_res_mlp,
            dec_use_res_mlp=dec_use_res_mlp,
            resmlp_kwargs=res_mlp_kwargs,
        )

        if self.action_network:
            self._action_embedding = nn.Linear(action_dim, embed_dim)

    def sample_prior(self, obs_dict=None, goal_dict=None, n=None):
        return self._vae.sample_prior(n=n, conditions=obs_dict, goals=goal_dict)

    def decode(self, obs_dict, goal_dict=None, z=None, n=None):
        conditions = {} # conditioned on the first image in a sequence
        for key in obs_dict:
            conditions[key] = obs_dict[key].expand(n, -1, -1)

        assert (n is not None) and (z is None)

        return self._vae.decode(conditions=conditions, goals=goal_dict, z=None, n=n)

    def decode_branched_future(self, obs_dict, goal_dict=None, z=None, n=None):
        conditions = {} # conditioned on the first image in a sequence
        for key in obs_dict:
            conditions[key] = obs_dict[key]
            
        assert (n is not None) and (z is None)

        return self._vae.decode(conditions=conditions, goals=goal_dict, z=None, n=n)

    def forward_train(self, inputs, conditions):

        return self._vae.forward(
            inputs=inputs,
            outputs=inputs,
            conditions=conditions,
            goals=None,
            freeze_encoder=False)

    def forward(self, next_embed, curr_embed, action):

        if self.action_network:
            action = self._action_embedding(action)

        return self.forward_train(
                inputs={"next_embed": next_embed},
                conditions={"curr_embed": curr_embed, "action": action},
            )

def classifier_large(encoder_output_size=512, output_size=1, dropout=None):
    return nn.Sequential(
        nn.Linear(encoder_output_size, 512),
        nn.LeakyReLU(),
        dropout,
        nn.Linear(512, 256),
        nn.LeakyReLU(),
        dropout,
        nn.Linear(256, output_size)
    )

def classifier_1(encoder_output_size=512, output_size=1, dropout=None):
    return nn.Sequential(
        nn.Linear(encoder_output_size, 256),
        nn.LeakyReLU(),
        dropout,
        nn.Linear(256, 128),
        nn.LeakyReLU(),
        dropout,
        nn.Linear(128, output_size)
    )

def classifier_2(encoder_output_size=512, output_size=1, dropout=None):
    return nn.Sequential(
        nn.Linear(encoder_output_size, 512),
        nn.LeakyReLU(),
        dropout,
        nn.Linear(512, output_size)
    )

def classifier_3(encoder_output_size=512, output_size=1, dropout=None):
    return nn.Linear(encoder_output_size, output_size)

class RewardClassifier(nn.Module):
    def __init__(self,
                 embed_dim,
                 lstm_config,
                 fc_num=0,
                 output_size=1,
                 ):
        super(RewardClassifier, self).__init__()

        self.lstm_hidden_size = lstm_config["hidden_size"]
        self.lstm_num_layers = lstm_config["num_layers"]
        self.lstm_bidirectional = lstm_config["bidirectional"]
        self.lstm_seq_length = lstm_config["seq_length"]    
        self.dropout_value = lstm_config["dropout"]
        self.dropout_layer = nn.Dropout(p=self.dropout_value)

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=self.lstm_hidden_size,
                            batch_first=True,
                            num_layers=self.lstm_num_layers,
                            dropout=self.dropout_value,
                            bidirectional=self.lstm_bidirectional)

        fc_input_size = self.lstm_hidden_size

        if fc_num == 0:
            self.last_fc = classifier_large(encoder_output_size=fc_input_size, 
                                            output_size=output_size, 
                                            dropout=self.dropout_layer)
        elif fc_num == 1:
            self.last_fc = classifier_1(encoder_output_size=fc_input_size, 
                                        output_size=output_size, 
                                        dropout=self.dropout_layer)  
        elif fc_num == 2:
            self.last_fc = classifier_2(encoder_output_size=fc_input_size,  
                                        output_size=output_size, 
                                        dropout=self.dropout_layer)
        elif fc_num == 3:
            self.last_fc = classifier_3(encoder_output_size=fc_input_size, 
                                        output_size=output_size, 
                                        dropout=self.dropout_layer)
        else:
            raise NotImplementedError
        
        self._num_directions = 2 if self.lstm_bidirectional else 1

        self._init_weights()

        # for evaluation
        self._rew_rnn_hidden_state = None
        self._rew_rnn_horizon = self.lstm_seq_length
        self._rnn_counter = 0

    def _init_weights(self):
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.kaiming_uniform_(p.weight)
                if p.bias is not None:
                    p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.kaiming_uniform_(p.weight, a=1.0)
                if p.bias is not None:
                    p.bias.data.zero_()

    def forward(self, features):
        logits_lst = []
        T = features.shape[1]
        for i in range(T - self.lstm_seq_length):
            logits = self.forward_one_step(features[:,i+1:i+1+self.lstm_seq_length,:]) # 9 step history + 1 step current
            logits_lst.append(logits)
        
        logits_lst = torch.stack(logits_lst, dim=1)
        return logits_lst

    def forward_one_step(self, features, return_seq=False):
        assert features.shape[1] == self.lstm_seq_length # for now to check
        lengths = torch.Tensor([self.lstm_seq_length] * features.shape[0])
        lengths = lengths.to('cpu')
        packed_features = nn.utils.rnn.pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        logits, _ = self.lstm(packed_features)
        logits, _ = nn.utils.rnn.pad_packed_sequence(logits, batch_first=True, padding_value=0)

        if not return_seq:
            if self.lstm_bidirectional:
                logits = logits[:, -1, :self.lstm_hidden_size] + logits[:, 0, self.lstm_hidden_size:]
            else:
                logits = logits[:, -1, :]
            logits = self.last_fc(logits)
        else:
            assert not self.lstm_bidirectional
            logits = TensorUtils.time_distributed(logits, self.last_fc)

        return logits

    def get_rnn_init_state(self, batch_size, device):
        """
        Get a default RNN state (zeros)
        Args:
            batch_size (int): batch size dimension

            device: device the hidden state should be sent to.

        Returns:
            hidden_state (torch.Tensor or tuple): returns tuple of hidden state tensors for RNN
        """
        h_0 = torch.zeros(self.lstm_num_layers * self._num_directions, batch_size, self.lstm_hidden_size).to(device)
        c_0 = torch.zeros(self.lstm_num_layers * self._num_directions, batch_size, self.lstm_hidden_size).to(device)
        return h_0, c_0
