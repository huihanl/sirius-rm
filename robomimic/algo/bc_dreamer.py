"""
Implementation of Behavioral Cloning (BC).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, res_mlp_args_from_config, res_mlp_args_from_config_vae, PolicyAlgo
from robomimic.models.base_nets import MLP, ResidualMLP

import torch.distributions as D

import robomimic.models.dyn_nets as DynNets
import numpy as np

from sklearn.metrics import confusion_matrix

from robomimic.algo import algo_factory
from robomimic.config import config_factory
import json

activation_map = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}

@register_algo_factory_func("bc_dreamer")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.
    Args:
        algo_config (Config instance): algo config
    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = ("gaussian" in algo_config and algo_config.gaussian.enabled)
    gmm_enabled = ("gmm" in algo_config and algo_config.gmm.enabled)
    vae_enabled = ("vae" in algo_config and algo_config.vae.enabled)

    if algo_config.dyn.combine_enabled:
        return BC_RNN_GMM_Dynamics_Combined, {}
    else:
        return BC_RNN_GMM_Dynamics_Seperate, {}

def dynamics_class(wm_config):
    if wm_config.dyn_class == "deter":
        return PolicyDynamics
    elif wm_config.dyn_class == "vae" and not wm_config.rew.lstm.enabled:
        return PolicyDynamicsVAE
    elif wm_config.dyn_class == "vae":
        return PolicyDynamicsVAE_RewardRNN
    else:
        raise NotImplementedError

class BC(PolicyAlgo):
    """
    Normal BC training.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **res_mlp_args_from_config(self.algo_config.res_mlp),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)  # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]

        if "hc_weights" in batch:
            input_batch["weights"] = batch["hc_weights"][:, 0]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping
            validate (bool): if True, don't perform any learning updates.
        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        actions = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        if "weights" in batch:
            assert NotImplementedError

        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.
        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
            max_grad_norm=self.algo_config.max_gradient_norm,
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BC, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)

class BC_RNN(BC):
    """
    BC training with an RNN policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **res_mlp_args_from_config(self.algo_config.res_mlp),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None)  # goals may not be present
        input_batch["actions"] = batch["actions"]

        input_batch["sparse_reward"] = batch.get("sparse_reward", None) 
        input_batch["dense_reward"] = batch.get("dense_reward", None) 
        input_batch["three_class"] = batch.get("three_class", None)
        input_batch["classifier_weights"] = batch.get("classifier_weights", None)

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)

        if "hc_weights" in batch:
            input_batch["weights"] = batch["hc_weights"]
        else:
            input_batch["weights"] = torch.ones_like(batch["action_modes"])

        """ Provide extra info for loss logging etc """
        if "intv_labels" in batch:
            input_batch["intv_labels"] = batch["intv_labels"]
        if "round" in batch:
            input_batch["round"] = batch["round"]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def get_action_with_dist(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state, dist = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state, return_dist=True)
        return action, dist

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0


class BC_RNN_GMM(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **res_mlp_args_from_config(self.algo_config.res_mlp),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)
        self._current_weights = None

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2  # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        res = OrderedDict()

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"]

        self._current_weights = batch["weights"]
        action_loss *= batch["weights"]
        action_loss = action_loss.mean()

        res["log_probs"] = -action_loss
        res["action_loss"] = action_loss

        return res

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        #log["Loss"] = info["losses"]["action_loss"].item()
        #log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]

        """ Log Category and Round Losses """
        if "category_loss" in info["losses"]:
            self._log_category_loss(log, info["losses"]["category_loss"])
        if "round_loss" in info["losses"]:
            self._log_round_loss(log, info["losses"]["round_loss"])
        if "category_count" in info["losses"]:
            self._log_category_count(log, info["losses"]["category_count"])

        return log

    def _log_data_attributes(self, log, key, entry):
        log[key + "/max"] = entry.max().item()
        log[key + "/min"] = entry.min().item()
        log[key + "/mean"] = entry.mean().item()
        log[key + "/std"] = entry.std().item()

    def _log_category_loss(self, log, cat_dict):
        for d in cat_dict:
            log["category_loss/" + d] = cat_dict[d].item()

    def _log_round_loss(self, log, round_dict):
        for d in round_dict:
            log["round_loss/" + d] = round_dict[d].item()

    def _log_category_count(self, log, cat_dict):
        for d in cat_dict:
            log["category_count/" + d] = cat_dict[d].item()


class BC_RNN_GMM_Dynamics(BC_RNN_GMM):
    """
    BC training with an RNN GMM policy and dynamics model.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled
        #assert self.algo_config.dynamics.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **res_mlp_args_from_config(self.algo_config.res_mlp),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        dyn_embed_dim = self.nets["policy"].nets["encoder"].output_shape()[0]
        
        self.nets["dynamics"] = DynNets.Dynamics(
            embed_dim=dyn_embed_dim,
            action_dim=self.ac_dim,
            hidden_dim=self.algo_config.dyn.hidden_dim,
            action_network_dim=self.algo_config.dyn.action_network_dim, 
            num_layers=self.algo_config.dyn.num_layers,
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)
        self._current_weights = None

    def _policy_forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists, obs_embedding = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            return_obs_embedding=True,
        )

        """ Policy Update """
        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2  # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
            obs_embedding=obs_embedding,
        )
        return predictions
    
    def _dynamics_forward_training(self, batch, obs_embedding):
        """ Dynamics Update """
        
        if self.algo_config.dyn.dyn_detach:
            obs_embedding = TensorUtils.detach(obs_embedding)

        dims_latent = list(range(len(obs_embedding.shape)))
        obs_embedding = torch.permute(obs_embedding, [1, 0] + dims_latent[2: ]) # (T, B, ...)

        dims_action = list(range(len(obs_embedding.shape)))
        actions = torch.permute(batch["actions"], [1, 0] + dims_action[2: ]) # (T, B, ...)

        pred_obs_embedding = self.nets["dynamics"](obs_embedding, actions)
        # Loss of next state prediction
        dyn_loss = nn.MSELoss()(obs_embedding[1:], pred_obs_embedding)

        predictions = OrderedDict(
            dyn_loss=dyn_loss,
        )

        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        res = OrderedDict()

        # loss is just negative log-likelihood of action targets
        policy_loss = -predictions["log_probs"]

        self._current_weights = batch["weights"]
        policy_loss *= batch["weights"]

        policy_loss = policy_loss.mean()

        dyn_loss = predictions["dyn_loss"].mean()

        res["log_probs"] = -policy_loss
        res["action_loss"] = policy_loss
        res["dyn_loss"] = dyn_loss

        return res
    
    def log_info(self, info):
        log = super(BC_RNN_GMM_Dynamics, self).log_info(info)

        log["loss/Dyn Loss"] = info["losses"]["dyn_loss"].item()
        log["loss/Policy Loss"] = info["losses"]["action_loss"].item()
        #log["loss/Total Loss"] = info["losses"]["action_loss"].item()

        # for wandb consistency
        log["policy/Policy Loss"] = info["losses"]["action_loss"].item()
        
        return log

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping
            validate (bool): if True, don't perform any learning updates.
        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)

            # policy forward pass
            policy_predictions = self._policy_forward_training(batch)

            # dynamics forward pass
            dyn_predictions = self._dynamics_forward_training(batch, 
                                                              policy_predictions["obs_embedding"])

            predictions = OrderedDict()
            predictions.update(policy_predictions)
            predictions.update(dyn_predictions)

            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(policy_predictions)
            info["losses"] = TensorUtils.detach(losses)

            policy_step_info = self._policy_train_step(losses)
            dyn_step_info = self._dyn_train_step(losses)
            info.update(policy_step_info)
            info.update(dyn_step_info)

        return info

    def _policy_train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.
        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
            max_grad_norm=self.algo_config.max_gradient_norm,
            retain_graph=not self.algo_config.dyn.dyn_detach,
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info
    
    def _dyn_train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.
        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["dynamics"],
            optim=self.optimizers["dynamics"],
            loss=losses["dyn_loss"],
            max_grad_norm=self.algo_config.max_gradient_norm,
        )
        info["dyn_grad_norms"] = policy_grad_norms
        return info
    

class BC_RNN_GMM_Dynamics_Combined(BC_RNN_GMM):
    """
    BC training with an RNN GMM policy and dynamics model.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled

        self.nets = nn.ModuleDict()

        self.dyn_class = dynamics_class(self.algo_config.dyn)
        self._batch_size = 16
        self._seq_length = 20
        self.nets["policy"] = self.dyn_class(self.obs_shapes,
                                        self.ac_dim,
                                        self.algo_config.dyn,
                                        self.algo_config,
                                        self.obs_config,
                                        self.goal_shapes,
                                        self.device,
                                        batch_size=self._batch_size,
                                        seq_length=self._seq_length,
                                        global_config=self.global_config
                                        )

        self.nets = self.nets.float().to(self.device)

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self._current_weights = None

    def _compute_losses(self, predictions, batch, epoch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        res = OrderedDict()

        total_loss = 0.

        # 1. policy loss
        if True: # self.algo_config.dyn.use_policy:
            res["log_probs"] = predictions["log_probs"].mean()
            policy_loss = -predictions["log_probs"]
            self._current_weights = batch["weights"][:,10:]
            policy_loss *= batch["weights"][:,10:]
            #assert (batch["weights"] == 1.).all()
            policy_loss = policy_loss.mean()
            res["action_loss"] = policy_loss
            
            if self.algo_config.dyn.use_policy:
                total_loss += policy_loss

        # 2. dynamics loss
        if True: #self.algo_config.dyn.use_dynamics:
            dyn_loss = predictions["dyn_loss"].mean()
            res["dyn_loss"] = dyn_loss

            if self.algo_config.dyn.use_dynamics and (self.algo_config.dyn.start_training_epoch is None or epoch >= self.algo_config.dyn.start_training_epoch):
                total_loss += dyn_loss * self.algo_config.dyn.dyn_weight
            
            # other losses
            if self.dyn_class is PolicyDynamicsVAE or self.dyn_class is PolicyDynamicsVAE_RewardRNN:
                res["kl_loss"] = predictions["kl_loss"].mean()
                res["recons_loss"] = predictions["recons_loss"].mean()

        # 3. reward loss
        if self.algo_config.dyn.use_reward:
            reward_loss = predictions["reward_loss"].mean()
            res["reward_loss"] = reward_loss
            total_loss += reward_loss * self.algo_config.dyn.reward_weight
        
        # 4. smoothness loss (optional)
        if self.algo_config.dyn.smooth_dynamics:
            smooth_loss = predictions["smooth_loss"].mean()
            res["smooth_loss"] = smooth_loss
            total_loss += smooth_loss * self.algo_config.dyn.smooth_weight
            
        # total loss
        res["total_loss"] = total_loss

        return res
    
    def log_info(self, info):
        log = super(BC_RNN_GMM_Dynamics_Combined, self).log_info(info)

        log["loss/Total Loss"] = info["losses"]["total_loss"].item()

        if True: #self.algo_config.dyn.use_policy:
            log["loss/Policy Loss"] = info["losses"]["action_loss"].item()

        if True: #self.algo_config.dyn.use_dynamics:
            log["loss/Dyn Loss"] = info["losses"]["dyn_loss"].item()
            if self.dyn_class is PolicyDynamicsVAE or self.dyn_class is PolicyDynamicsVAE_RewardRNN:
                log["loss/Dyn_Recons"] = info["losses"]["recons_loss"].item()
                log["loss/Dyn_KL"] = info["losses"]["kl_loss"].item()

        if self.algo_config.dyn.use_reward:
            log["loss/Reward Loss"] = info["losses"]["reward_loss"].item()
            log["loss/reward/acc all"] = info["predictions"]["reward_overal_acc"].item()
            log["loss/reward/acc class 0"] = info["predictions"]["reward_class0_acc"].item()
            log["loss/reward/acc class 1"] = info["predictions"]["reward_class1_acc"].item()
            if "reward_class2_acc" in info["predictions"]:
                log["loss/reward/acc class 2"] = info["predictions"]["reward_class2_acc"].item()
            if "confusion_matrix" in info["predictions"]:
                log["confusion_matrix"] = info["predictions"]["confusion_matrix"]

        if self.algo_config.dyn.smooth_dynamics:
            log["loss/Smooth Loss"] = info["losses"]["smooth_loss"].item()

        if self.algo_config.dyn.stochastic_inputs:
            self._log_stoch_inputs(log, info)

        if self._current_weights is not None:
            self._log_data_attributes(log, "weights", self._current_weights)

        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]

        return log

    def _log_stoch_inputs(self, log, info):
        mean, std = info["predictions"]["obs_embedding"].chunk(2, -1)

        min_std = 0.1
        max_std = 2.0
        std = max_std * torch.sigmoid(std) + min_std

        self._log_data_attributes(log, "embedding_mean", mean)
        self._log_data_attributes(log, "embedding_std", std)
        
        ####
        mean, std = info["predictions"]["pred_obs_embedding"].chunk(2, -1)

        min_std = 0.1
        max_std = 2.0
        std = max_std * torch.sigmoid(std) + min_std

        self._log_data_attributes(log, "pred_embedding_mean", mean)
        self._log_data_attributes(log, "pred_embedding_std", std)

    def _log_data_attributes(self, log, key, entry):
        log[key + "/max"] = entry.max().item()
        log[key + "/min"] = entry.min().item()
        log[key + "/mean"] = entry.mean().item()
        log[key + "/std"] = entry.std().item()

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping
            validate (bool): if True, don't perform any learning updates.
        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)

            # policy forward pass
            predictions = self.nets["policy"].forward_train(batch)

            losses = self._compute_losses(predictions, batch, epoch)
            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                policy_step_info = self._policy_train_step(losses)
                info.update(policy_step_info)

        return info

    def _policy_train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.
        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["total_loss"],
            max_grad_norm=self.algo_config.max_gradient_norm,
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state, obs_embedding = self.nets["policy"].nets["policy"].forward_step(obs_to_use, 
                                                                                                        goal_dict=goal_dict, 
                                                                                                        rnn_state=self._rnn_hidden_state,
                                                                                                        return_obs_embedding=True)
        
        self.obs_embedding_cache.append(obs_embedding)
        self.action_cache.append(action)

        if len(self.obs_embedding_cache) > self._seq_length:
            self.obs_embedding_cache = self.obs_embedding_cache[-self._seq_length:]
            assert len(self.obs_embedding_cache) == self._seq_length # hardcode for now

        if len(self.action_cache) > self._seq_length:
            self.action_cache = self.action_cache[-self._seq_length:]
            assert len(self.action_cache) == self._seq_length # hardcode for now

        return action, obs_embedding

    def get_obs_embedding(self, obs_dict):
        assert not self.nets.training
        for k in obs_dict:
            obs_dict[k] = torch.unsqueeze(obs_dict[k], 1)
        return self.nets["policy"].nets["policy"].forward_embedding_only(obs_dict)
    
    def imagine(self, obs_dict, goal_dict=None, horizon=10, n_futures=200):
        """ Imagine the next steps from obs_dict, given the past history stored in self.obs_embedding_cache """
        
        with torch.no_grad():
            # Imagine the next steps with horizon
            assert not self.nets.training

            # reset rnn hidden state for imagining rollout, to not interfere with the one for the policy
            self._img_rnn_hidden_state = None

            for k in obs_dict:
                obs_dict[k] = torch.tile(obs_dict[k], [n_futures] + [1] * len(obs_dict[k].shape[1:]))

            if self._img_rnn_hidden_state is None:
                batch_size = n_futures #list(obs_dict.values())[0].shape[0]
                self._img_rnn_hidden_state = self.nets["policy"].nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)
            
            curr_embed = self.get_obs_embedding(obs_dict)

            if len(self.obs_embedding_cache) < self._seq_length // 2: # hardcode for now
                return np.zeros([n_futures, horizon, curr_embed.shape[-1]])
            
            imagined_embeddings = []
            imagined_actions = []
            
            _copy_obs_embedding_cache = torch.tile(torch.cat(self.obs_embedding_cache, dim=1)[:, -horizon :, :], (n_futures, 1, 1))
            for i in range(horizon):
                action, self._img_rnn_hidden_state = self.nets["policy"].nets["policy"].forward_policy_only(obs_embedding=curr_embed, 
                                                                                        rnn_init_state=self._img_rnn_hidden_state) # action shape: (n, 7), hidden shape: ([2, n, 1000], [2, n, 1000])

                if len(imagined_embeddings) == 0:
                    curr_input = _copy_obs_embedding_cache.clone().detach()
                else:
                    curr_input = torch.cat((_copy_obs_embedding_cache[:, i:, :], torch.cat(imagined_embeddings, dim=1)), dim=1)

                curr_input = curr_input.reshape(-1, curr_input.shape[1] * curr_input.shape[2])
                action_embedding = self.nets["policy"].nets["dynamics"].cell._action_embedding(action)

                obs_dict = {"curr_embed": curr_input, "action": action_embedding}
                curr_embed = self.nets["policy"].nets["dynamics"].cell.decode_branched_future(obs_dict=obs_dict, n=n_futures)['next_embed'][:, None, :] # iterative

                imagined_embeddings.append(curr_embed)
                imagined_actions.append(action)
                
            self.imagined_embeddings = imagined_embeddings
            self.imagined_actions = imagined_actions

            imagined_embeddings = TensorUtils.to_numpy(imagined_embeddings)
            imagined_actions = TensorUtils.to_numpy(imagined_actions)
            
            return np.concatenate(imagined_embeddings, axis=1) #, np.array(imagined_actions)
    
    def compute_reward_test_time(self, imagine="none", n_futures=200):

        reward_obs_length = self.algo_config.dyn.rew.lstm.seq_length # ==10
        
        if imagine == "none": # This option is not supported right now
            assert False, "\"none\" option is not supported right now"
            if len(self.obs_embedding_cache) < reward_obs_length:
                return 0, 1
            obs_embedding = torch.cat(self.obs_embedding_cache[-reward_obs_length:], dim=0)
            actions = torch.cat(self.action_cache[-reward_obs_length:], dim=0)
        elif imagine == "last":
            if self.imagined_embeddings is None:
                pred_label, pred_prob = np.zeros((n_futures,)), np.zeros((n_futures, 3))
                pred_prob[:, 0] = 1
                return pred_label, pred_prob

            obs_embedding = torch.cat(self.imagined_embeddings, dim=1)
            actions = torch.stack(self.imagined_actions, dim=1)
        elif imagine == "all":
            if self.imagined_embeddings is None:
                pred_label, pred_prob = np.zeros((n_futures, reward_obs_length)), np.zeros((n_futures, reward_obs_length, 3))
                pred_prob[:, :, 0] = 1
                return pred_label, pred_prob

            imagined_cat = torch.cat(self.imagined_embeddings, dim=1)
            obs_cache_cat = torch.tile(torch.cat(self.obs_embedding_cache[-reward_obs_length:], dim=1), (imagined_cat.shape[0], 1, 1))
            obs_embedding = torch.cat((obs_cache_cat[:, -reward_obs_length:, :], imagined_cat), dim=1)
            
            imagined_actions_cat = torch.stack(self.imagined_actions, dim=1)
            actions_cache_cat = torch.tile(torch.stack(self.action_cache, dim=1), (imagined_actions_cat.shape[0], 1, 1))
            actions = torch.cat((actions_cache_cat[:, -reward_obs_length:, :], imagined_actions_cat), dim=1)
        else:
            raise NotImplementedError

        assert obs_embedding.shape[0] == actions.shape[0]
        assert len(obs_embedding.shape) == len(actions.shape)

        rew_net = self.nets["policy"].nets["reward"]

        if self.nets["policy"].wm_configs.rew.use_action:
            # R(s_t, a_t)
            obs_action_embedding = torch.cat([obs_embedding, actions], dim=-1)
        
        if imagine == "all":
            pred_reward = rew_net.forward(obs_action_embedding)
            pred_label = torch.argmax(pred_reward, dim=-1)
            pred_prob = nn.Softmax(dim=-1)(pred_reward) # prob of failure
        else:
            pred_reward = rew_net.forward_one_step(obs_action_embedding)
            pred_label = torch.argmax(pred_reward, dim=-1)
            pred_prob = nn.Softmax(dim=-1)(pred_reward) # prob of failure
            
        pred_label = TensorUtils.to_numpy(pred_label)
        pred_prob = TensorUtils.to_numpy(pred_prob)
        return pred_label, pred_prob

    def reset(self, eval_policy_only=True):
        super(BC_RNN_GMM_Dynamics_Combined, self).reset()
        """ 
        eval_policy_only: when we only evaluate policy success rate, no dynamics 
        """
        if not self.nets.training and eval_policy_only:
            self.nets["policy"].nets["dynamics"] = self.nets["policy"].nets["dynamics"].cpu()

        self.obs_embedding_cache = []
        self.action_cache = []
        
        self.imagined_embeddings = None
        self.imagined_actions = None

    def reset_history_only(self):
       
        self.obs_embedding_cache = []

        self.action_cache = []
        
        self.imagined_embeddings = None
        self.imagined_actions = None



class BC_RNN_GMM_Dynamics_Seperate(BC_RNN_GMM_Dynamics_Combined):
    
    def _create_optimizers(self):
        """
        Override default optimizers since policy and dynamics are seperate
        """
        self.optimizers = dict()
        self.lr_schedulers = dict()

        for k in self.optim_params:
            # only make optimizers for networks that have been created - @optim_params may have more
            # settings for unused networks
            if k in self.nets["policy"].nets.keys():
                if isinstance(self.nets["policy"].nets[k], nn.ModuleList):
                    self.optimizers[k] = [
                        TorchUtils.optimizer_from_optim_params(net_optim_params=self.optim_params[k], net=self.nets["policy"].nets[k][i])
                        for i in range(len(self.nets["policy"].nets[k]))
                    ]
                    self.lr_schedulers[k] = [
                        TorchUtils.lr_scheduler_from_optim_params(net_optim_params=self.optim_params[k], net=self.nets["policy"].nets[k][i], optimizer=self.optimizers[k][i])
                        for i in range(len(self.nets["policy"].nets[k]))
                    ]
                else:
                    self.optimizers[k] = TorchUtils.optimizer_from_optim_params(
                        net_optim_params=self.optim_params[k], net=self.nets["policy"].nets[k])
                    self.lr_schedulers[k] = TorchUtils.lr_scheduler_from_optim_params(
                        net_optim_params=self.optim_params[k], net=self.nets["policy"].nets[k], optimizer=self.optimizers[k])
    
    
    # newer version of policy train step that uses seperate optimizers for policy and dynamics
    def _policy_train_step(self, losses):
        info = OrderedDict()
        
        for k in self.optimizers:
            self.optimizers[k].zero_grad()
        losses['total_loss'].backward(retain_graph=False)

        # gradient clipping
        max_grad_norm = self.algo_config.max_gradient_norm
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.nets.parameters(), max_grad_norm)
            
        # compute grad norms
        grad_norms = 0.
        for p in self.nets.parameters():
            # only clip gradients for parameters for which requires_grad is True
            if p.grad is not None:
                grad_norms += p.grad.data.norm(2).pow(2).item()

        # step
        for k in self.optimizers:
            self.optimizers[k].step()
        
        info["policy_grad_norms"] = grad_norms
        return info
    
class PolicyDynamics(nn.Module):

    def __init__(self,
                 obs_shapes,
                 ac_dim,
                 wm_configs,
                 algo_config,
                 obs_config,
                 goal_shapes,
                 device,
                 batch_size,
                 seq_length,
                 global_config=None,
                 ):

        super(PolicyDynamics, self).__init__()

        self.action_dim = ac_dim
        self.ac_dim = ac_dim

        self.wm_configs = wm_configs
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.obs_shapes = obs_shapes
        self.goal_shapes = goal_shapes
        self.device = device 

        self._batch_size = batch_size
        self._seq_length = seq_length

        self.global_config = global_config

        self.stochastic_inputs = self.wm_configs.stochastic_inputs
        self.kl_balance = self.wm_configs.kl_balance

        try:
            self.load_prev_policy_dyn = self.wm_configs.load_prev_policy_dyn
            self.load_ckpt = self.wm_configs.load_ckpt

            import os
            dir_name = os.path.dirname(os.path.dirname(self.load_ckpt)) 
            load_config = os.path.join(dir_name, "config.json")
            ext_cfg = json.load(open(load_config, 'r'))
            config = config_factory(ext_cfg["algo_name"])
            with config.values_unlocked():
                config.update(ext_cfg)
            self.load_config = config
        except:
            self.load_prev_policy_dyn = False
            self.load_ckpt = None
            self.load_config = None

        self.nets = nn.ModuleDict()

        self.create_policy()

        dyn_embed_dim = self.nets["policy"].nets["encoder"].output_shape()[0]
        if self.stochastic_inputs:
            dyn_embed_dim *= 2

        self.create_dynamics(dyn_embed_dim)

        if self.wm_configs.get("use_reward", False):
            self.create_reward(dyn_embed_dim)

        self.nets = self.nets.float().to(self.device)

    def create_policy(self):

        if self.load_prev_policy_dyn:

            model = algo_factory(
                algo_name=self.global_config.algo_name,
                config=self.load_config,
                obs_key_shapes=self.obs_shapes,
                ac_dim=self.ac_dim,
                device=self.device,
            )
            
            print('Loading model weights from:', self.load_ckpt)
            
            from robomimic.utils.file_utils import maybe_dict_from_checkpoint
            ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=self.load_ckpt)

            model.deserialize(ckpt_dict["model"])

            self.nets["policy"] = model.nets["policy"].nets["policy"]
            return

        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            stochastic_inputs=self.stochastic_inputs,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **res_mlp_args_from_config(self.algo_config.res_mlp),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

    def create_dynamics(self, dyn_embed_dim):

        if self.load_prev_policy_dyn:
            model = algo_factory(
                algo_name=self.global_config.algo_name,
                config=self.load_config,
                obs_key_shapes=self.obs_shapes,
                ac_dim=self.ac_dim,
                device=self.device,
            )
            print('Loading model weights from:', self.load_ckpt)
            
            from robomimic.utils.file_utils import maybe_dict_from_checkpoint
            ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=self.load_ckpt)
            model.deserialize(ckpt_dict["model"])

            self.nets["dynamics"] = model.nets["policy"].nets["dynamics"]
            return

        self.nets["dynamics"] = DynNets.Dynamics(
            embed_dim=dyn_embed_dim,
            action_dim=self.ac_dim,
            hidden_dim=self.wm_configs.hidden_dim,
            action_network_dim=self.wm_configs.action_network_dim, 
            num_layers=self.wm_configs.num_layers,
            use_res_mlp=self.wm_configs.use_res_mlp,
        )

    def create_reward(self, dyn_embed_dim):

        rew = self.wm_configs.rew
        if rew.get("use_action", True): 
            input_dim = dyn_embed_dim + self.ac_dim
        else:
            input_dim = dyn_embed_dim

        if rew.use_res_mlp:
            hidden_dim = rew.hidden_dim
            self.nets["reward"] = nn.Sequential(
                ResidualMLP(
                    input_dim=input_dim,
                    activation=nn.ReLU,
                    output_activation=None,
                    num_blocks=4,
                    hidden_dim=hidden_dim,
                    normalization=True
                ),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),
            )
        else:
            self.nets["reward"] = MLP(
                    input_dim=input_dim,
                    output_dim=1,
                    layer_dims=[rew.hidden_dim] * rew.num_layers,
                    activation=activation_map[rew.activation],
                    output_activation=nn.Sigmoid,
                    normalization=True,
                )

        self._create_reward_loss(rew)

    def _create_reward_loss(self, rew):

        if rew.rew_class == "binary" and rew.binary_loss == "bce":
            self._reward_loss = nn.BCEWithLogitsLoss(reduction="none")
        elif rew.rew_class == "binary" and rew.binary_loss == "focal":
            raise NotImplementedError
        elif rew.rew_class == "three_class":
            self._reward_loss = nn.CrossEntropyLoss(reduction="none")
        else:
            self._reward_loss = nn.MSELoss(reduction="none")

    def kl_loss(self, pred, actual):
        d = self.zdistr
        d_pred, d_actual = d(pred), d(actual)
        loss_kl_exact = D.kl.kl_divergence(d_actual, d_pred)  # (T,B,I)

        # Analytic KL loss, standard for VAE
        if self.kl_balance < 0:
            loss_kl = loss_kl_exact
        else:
            actual_to_pred = D.kl.kl_divergence(d_actual, d(TensorUtils.detach(pred)))
            pred_to_actual = D.kl.kl_divergence(d(TensorUtils.detach(actual)), d_pred)

            loss_kl = (1 - self.kl_balance) * actual_to_pred + self.kl_balance * pred_to_actual

        # average across batch
        return loss_kl.mean()

    def zdistr(self, z):
        return self.diag_normal(z)

    def diag_normal(self, z, min_std=0.1, max_std=2.0):
        mean, std = z.chunk(2, -1)
        std = max_std * torch.sigmoid(std) + min_std
        return D.independent.Independent(D.normal.Normal(mean, std), 1)

    def forward_train(self, batch):
        raise NotImplementedError


class PolicyDynamicsVAE(PolicyDynamics):

    def __init__(self,
                 obs_shapes,
                 ac_dim,
                 wm_configs,
                 algo_config,
                 obs_config,
                 goal_shapes,
                 device,
                 batch_size,
                 seq_length,
                 global_config=None,
                 ):

        super(PolicyDynamicsVAE, self).__init__(
                 obs_shapes,
                 ac_dim,
                 wm_configs,
                 algo_config,
                 obs_config,
                 goal_shapes,
                 device,
                 batch_size,
                 seq_length,
                 global_config
                 )

    def create_dynamics(self, dyn_embed_dim):

        if self.load_prev_policy_dyn:
            model = algo_factory(
                algo_name=self.global_config.algo_name,
                config=self.load_config,
                obs_key_shapes=self.obs_shapes,
                ac_dim=self.ac_dim,
                device=self.device,
            )
            print('Loading model weights from:', self.load_ckpt)

            from robomimic.utils.file_utils import maybe_dict_from_checkpoint
            ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=self.load_ckpt)
            model.deserialize(ckpt_dict["model"])

            self.nets["dynamics"] = model.nets["policy"].nets["dynamics"]
            return

        self.nets["dynamics"] = DynNets.DynamicsVAE(
            embed_dim=dyn_embed_dim,
            action_dim=self.ac_dim,
            device=self.device,
            algo_config=self.algo_config,
            obs_config=self.obs_config,
            use_history=self.wm_configs.use_history,
            use_real=self.wm_configs.use_real,
        )

    def _half_batch(self, batch, idx):
        # Split batch into two halves
        if idx == 0:
            start, end = 0, batch["actions"].shape[1] // 2
        else:
            start, end = batch["actions"].shape[1] // 2, batch["actions"].shape[1]

        batch_half = {}
        for key, value in batch.items():
            if value is None:
                batch_half[key] = value
                continue
            if isinstance(value, dict):
                batch_half[key] = {}
                for key2, value2 in value.items():
                    batch_half[key][key2] = value2[:, start:end]
            else:
                if key == "classifier_weights":
                    batch_half[key] = value[:]
                else:
                    batch_half[key] = value[:, start:end]
        
        assert batch_half["actions"].shape[0] == self._batch_size
        assert batch_half["actions"].shape[1] == self._seq_length // 2
                
        return batch_half

    def forward_train(self, batch):
        # batch: (B, T, ...)
        assert batch["actions"].shape[1] == self._seq_length

        original_batch = batch

        batch_first_half = self._half_batch(batch, 0)
        batch_second_half = self._half_batch(batch, 1)

        assert batch_first_half["actions"].shape[1] == self._seq_length // 2
        assert batch_second_half["actions"].shape[1] == self._seq_length // 2

        obs_embedding_first_half = self.nets["policy"].forward_embedding_only(obs_dict=batch_first_half["obs"])
        obs_embedding_second_half = self.nets["policy"].forward_embedding_only(obs_dict=batch_second_half["obs"])
        
        if self.algo_config.dyn.obs_sg:
            obs_embedding_first_half = TensorUtils.detach(obs_embedding_first_half)
            obs_embedding_second_half = TensorUtils.detach(obs_embedding_second_half)

        predictions = OrderedDict()

        # if self.algo_config.dyn.load_prev_policy_dyn True, still compute value for sanity check
        if True: #self.algo_config.dyn.use_policy:
            log_probs = self._compute_policy(batch=batch_second_half, 
                                             obs_embedding=obs_embedding_second_half)
            predictions.update(log_probs=log_probs)

        if True: #self.algo_config.dyn.use_dynamics:
            dyn_predictions = self._compute_dynamics(obs_embedding_first_half=obs_embedding_first_half, 
                                                     obs_embedding=obs_embedding_second_half, 
                                                     batch=batch_second_half)
            predictions.update(dyn_predictions)

        if self.algo_config.dyn.use_reward:
            self._compute_reward(batch=batch_second_half, 
                                 original_batch=original_batch, 
                                 obs_embedding_first_half=obs_embedding_first_half, 
                                 obs_embedding=obs_embedding_second_half, 
                                 predictions=predictions)

        return predictions
    
    def _compute_policy(self, batch, obs_embedding):
        """ Policy Update (B, T, ...) """
        dists = self.nets["policy"].forward_policy_only_train(obs_embedding)
        assert len(dists.batch_shape) == 2  # [B, T]
        log_probs = dists.log_prob(batch["actions"])
        assert obs_embedding.shape[1] == self._seq_length // 2 
        assert batch["actions"].shape[1] == self._seq_length // 2
        return log_probs
    
    def _compute_dynamics(self, obs_embedding_first_half, obs_embedding, batch):
        """ Dynamics Update (B, T, ...) """
        assert obs_embedding_first_half.shape[1] == self._seq_length // 2 # (B, T, ...)
        assert obs_embedding.shape[1] == self._seq_length // 2  # (B, T, ...)
        assert batch["actions"].shape[1] == self._seq_length // 2 # (B, T, ...)

        assert obs_embedding_first_half.shape[0] == self._batch_size # (B, T, ...)
        assert obs_embedding.shape[0] == self._batch_size # (B, T, ...)
        assert batch["actions"].shape[0] == self._batch_size # (B, T, ...)

        # Concat history embedding
        obs_embedding = torch.cat([obs_embedding_first_half, obs_embedding], dim=1)
        dims_latent = list(range(len(obs_embedding.shape)))
        obs_embedding = torch.permute(obs_embedding, [1, 0] + dims_latent[2: ]) # (T, B, ...)
        dims_action = list(range(len(obs_embedding.shape)))
        actions = torch.permute(batch["actions"], [1, 0] + dims_action[2: ]) # (T, B, ...)

        assert obs_embedding.shape[0] == self._seq_length and actions.shape[0] == self._seq_length // 2

        predictions = self.nets["dynamics"](obs_embedding, actions)

        return predictions

    def _compute_reward(self, batch, original_batch, obs_embedding_first_half, obs_embedding, predictions):
        raise NotImplementedError
    
    def _confusion_matrix(self, y_true, y_pred, num_classes):
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        for t, p in zip(y_true.view(-1), y_pred.view(-1)):
            conf_matrix[t.long(), p.long()] += 1
        conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix_no_nan = torch.where(torch.isnan(conf_matrix_norm), torch.zeros_like(conf_matrix_norm), conf_matrix_norm)
        return torch.diag(conf_matrix_no_nan), conf_matrix_no_nan

class PolicyDynamicsVAE_RewardRNN(PolicyDynamicsVAE):

    def create_reward(self, dyn_embed_dim):

        rew = self.wm_configs.rew
        assert rew.lstm.enabled

        if rew.get("use_action", True): 
            input_dim = dyn_embed_dim + self.ac_dim
        else:
            input_dim = dyn_embed_dim

        self.nets["reward"] = DynNets.RewardClassifier(
            embed_dim=input_dim,
            lstm_config=self.algo_config.dyn.rew.lstm,
            fc_num=rew.fc_num,
            output_size=3 if self.algo_config.dyn.rew.rew_class == "three_class" else 1
        )

        self._lstm_seq_length = self.algo_config.dyn.rew.lstm.seq_length
        assert self._lstm_seq_length <= self._seq_length

        self._create_reward_loss(rew)

    def _compute_reward(self, batch, original_batch, obs_embedding_first_half, obs_embedding, predictions):

        assert self.algo_config.dyn.rew.rew_class in ["binary", "three_class"] # for now
        obs_embedding = torch.cat([obs_embedding_first_half, obs_embedding], dim=1)
        
        actions = original_batch["actions"]
        assert obs_embedding.shape[0] == actions.shape[0]

        # reward prediction
        if not self.algo_config.dyn.rew.all_seq_prediction:
            # normal, one sequence - one reward label
            if self.algo_config.dyn.rew.rew_class == "binary":
                reward_data = original_batch["sparse_reward"][:,-1:] * -1. 
            elif self.algo_config.dyn.rew.rew_class == "three_class":
                reward_data = original_batch["three_class"][:,-1:] * -1.
            else:
                reward_data = original_batch["dense_reward"][:,-1:]

            if self.wm_configs.rew.use_action:
                # R(s_t, a_t)
                obs_action_embedding = torch.cat([obs_embedding, actions], dim=-1)
                pred_reward = self.nets["reward"].forward_one_step(obs_action_embedding[:,-self._lstm_seq_length:,:])
            else:
                # R(s_t)
                pred_reward = self.nets["reward"].forward_one_step(obs_embedding[:,-self._lstm_seq_length:,:])
        else:
            # "rnn" prediction: one sequence - one sequence of labels
            if self.algo_config.dyn.rew.rew_class == "binary":
                reward_data = original_batch["sparse_reward"][:,-self._lstm_seq_length:] * -1. 
            elif self.algo_config.dyn.rew.rew_class == "three_class":
                reward_data = original_batch["three_class"][:,-self._lstm_seq_length:] * -1.
            else:
                reward_data = original_batch["dense_reward"][:,-self._lstm_seq_length:] * -1

            if self.wm_configs.rew.use_action:
                # R(s_t, a_t)
                obs_action_embedding = torch.cat([obs_embedding, actions], dim=-1)
                pred_reward = self.nets["reward"].forward_one_step(obs_action_embedding[:,-self._lstm_seq_length:,:], return_seq=True)
            else:
                # R(s_t)
                pred_reward = self.nets["reward"].forward_one_step(obs_embedding[:,-self._lstm_seq_length:,:], return_seq=True)

        """ compute reward loss """ 
        if self.algo_config.dyn.rew.rew_class != "three_class":
            assert pred_reward.shape == reward_data.shape
            pred_reward = torch.flatten(pred_reward)
            reward_data = torch.flatten(reward_data)
        else:
            reward_data = torch.flatten(reward_data)
            reward_data = torch.tensor(reward_data, dtype=torch.long)
            pred_reward = pred_reward.view(-1, pred_reward.size(-1))

        reward_loss = self._reward_loss(pred_reward, reward_data)
        
        if self.algo_config.dyn.rew.use_weighted_loss:
            class_weight = batch["classifier_weights"]
            assert class_weight is not None and class_weight.shape == reward_loss.shape
            reward_loss = (class_weight * reward_loss).mean()
        else:
            reward_loss = reward_loss.mean()
        predictions["reward_loss"] = reward_loss

        """ logging """
        if self.algo_config.dyn.rew.rew_class == "binary":
            # reward accuracy
            threshold = 0.5
            pred_labels = pred_reward > threshold
            overal_acc = (pred_labels == reward_data).float().mean()

            with np.errstate(divide='ignore', invalid='ignore'):
                class_acc, matrix = self._confusion_matrix(y_true=reward_data, y_pred=pred_labels, num_classes=2)

            predictions["reward_overal_acc"] = overal_acc
            predictions["reward_class0_acc"] = class_acc[0]
            predictions["reward_class1_acc"] = class_acc[1]
            predictions["confusion_matrix"] = matrix

        elif self.algo_config.dyn.rew.rew_class == "three_class":
            # reward accuracy
            pred_labels = torch.argmax(pred_reward, dim=-1)
            overal_acc = (pred_labels == reward_data).float().mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                class_acc, matrix = self._confusion_matrix(y_true=reward_data, y_pred=pred_labels, num_classes=3)
            
            predictions["reward_overal_acc"] = overal_acc
            predictions["reward_class0_acc"] = class_acc[0] # normal
            predictions["reward_class1_acc"] = class_acc[1] # normal
            predictions["reward_class2_acc"] = class_acc[2] # normal
            predictions["confusion_matrix"] = matrix

