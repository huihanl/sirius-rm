import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.value_nets as ValueNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import register_algo_factory_func, res_mlp_args_from_config, ValueAlgo, PolicyAlgo


@register_algo_factory_func("iql")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the AWAC algo class to instantiate, along with additional algo kwargs.
    Args:
        algo_config (Config instance): algo config
    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return IQL, {}


class IQL(PolicyAlgo, ValueAlgo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.td_loss_fcn = nn.SmoothL1Loss() if self.algo_config.critic.use_huber else nn.MSELoss()

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        Networks for this algo: critic (potentially ensemble), policy
        """

        # Create nets
        self.nets = nn.ModuleDict()

        # Assemble args to pass to actor
        actor_args = dict(self.algo_config.actor.net.common)

        # Add network-specific args and define network class
        if self.algo_config.actor.net.type == "gaussian":
            actor_cls = PolicyNets.GaussianActorNetwork
            actor_args.update(dict(self.algo_config.actor.net.gaussian))
        elif self.algo_config.actor.net.type == "gmm":
            actor_cls = PolicyNets.GMMActorNetwork
            actor_args.update(dict(self.algo_config.actor.net.gmm))
        else:
            # Unsupported actor type!
            raise ValueError(f"Unsupported actor requested. "
                             f"Requested: {self.algo_config.actor.net.type}, "
                             f"valid options are: {['gaussian']}")

        # Policy
        self.nets["actor"] = actor_cls(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor.layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **res_mlp_args_from_config(self.algo_config.actor.res_mlp),
            **actor_args,
        )

        # Critics
        self.nets["critic"] = nn.ModuleList()
        self.nets["critic_target"] = nn.ModuleList()
        for _ in range(self.algo_config.critic.ensemble.n):
            for net_list in (self.nets["critic"], self.nets["critic_target"]):
                critic = ValueNets.ActionValueNetwork(
                    obs_shapes=self.obs_shapes,
                    ac_dim=self.ac_dim,
                    mlp_layer_dims=self.algo_config.critic.layer_dims,
                    goal_shapes=self.goal_shapes,
                    encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
                    **res_mlp_args_from_config(self.algo_config.critic.res_mlp),
                )
                net_list.append(critic)

        self.nets["vf"] = ValueNets.ValueNetwork(
            obs_shapes=self.obs_shapes,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **res_mlp_args_from_config(self.algo_config.critic.res_mlp),
        )

        # Send networks to appropriate device
        self.nets = self.nets.float().to(self.device)

        # sync target networks at beginning of training
        with torch.no_grad():
            for critic, critic_target in zip(self.nets["critic"], self.nets["critic_target"]):
                TorchUtils.hard_update(
                    source=critic,
                    target=critic_target,
                )

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out relevant info and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()

        # remove temporal batches for all
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["next_obs"] = {k: batch["obs"][k][:, 1, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = torch.clamp(batch["actions"][:, 0, :], min=-1, max=1)

        bs = len(input_batch["actions"])

        # relabel dones
        relabel_dones_mode = self.algo_config.relabel_dones_mode
        if relabel_dones_mode is None:
            input_batch["dones"] = batch["dones"][:, 0]
        elif relabel_dones_mode == 'intv':
            input_batch["dones"] = ((batch["action_modes"][:,0] == 1) * 1.0) # done wherever intv
        else:
            raise ValueError

        if self.algo_config.ignore_dones:
            input_batch["dones"] = torch.zeros(bs)


        # relabel rewards
        relabel_rewards_mode = self.algo_config.relabel_rewards_mode
        if relabel_rewards_mode is None:
            input_batch["rewards"] = batch["rewards"][:, 0]
        elif relabel_rewards_mode == 'avoid_intv':
            input_batch["rewards"] = torch.ones(bs)
            input_batch["rewards"] -= ((batch["action_modes"][:,0] == 1) * 1.0) # reward is decremented wherever there is intv
        else:
            raise ValueError

        if self.algo_config.use_negative_rewards:
            input_batch["rewards"] = input_batch["rewards"] - torch.ones(len(input_batch["rewards"]))

        if "final_success" in batch.keys():
            # action_modes and final_success masks
            input_batch["final_success"] = batch["final_success"][:, 0]
        if "action_modes" in batch.keys():
            input_batch["action_modes"] = batch["action_modes"][:, 0]

        if self.algo_config.use_shaped_rewards:
            raise NotImplementedError

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
        info = OrderedDict()

        # Set the correct context for this training step
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # Always run super call first
            info = super().train_on_batch(batch, epoch, validate=validate)

            # Train critic(s)
            critic_info, critic_losses, vf_loss = self._compute_critic_loss(batch, epoch)
            # Train actor
            actor_info, policy_loss = self._compute_policy_loss(batch, critic_info, epoch)

            # Critic update
            self._update_critic(critic_losses, vf_loss, validate)
            # Actor update
            self._update_policy(policy_loss, validate)

            # Update info
            info.update(actor_info)
            info.update(critic_info)

        # Return stats
        return info

    def _compute_policy_loss(self, batch, critic_info, epoch):
        info = OrderedDict()

        dist = self.nets["actor"].forward_train(obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        log_prob = dist.log_prob(batch["actions"])

        info["actor/log_prob"] = log_prob.mean()

        q_pred = critic_info["vf/q_pred"]
        v_pred = critic_info["vf/v_pred"]
        adv = q_pred - v_pred
        weights, adv_clipped = self._apply_adv_filter(adv)

        policy_loss = (-log_prob * weights.detach()).mean()

        info["actor/loss"] = policy_loss

        # log adv-related values
        info["adv/adv_raw"] = adv
        info["adv/adv_score"] = adv_clipped
        info["adv/adv_weight"] = weights
        info["adv/V_value"] = v_pred

        # Return stats
        return info, policy_loss

    def _update_policy(self, policy_loss, validate=False):
        if validate:
            return

        TorchUtils.backprop_for_loss(
            net=self.nets["actor"],
            optim=self.optimizers["actor"],
            loss=policy_loss,
            max_grad_norm=self.algo_config.actor.max_gradient_norm,
        )

    def _compute_critic_loss(self, batch, epoch):
        info = OrderedDict()

        rewards = torch.unsqueeze(batch["rewards"], 1)
        dones = torch.unsqueeze(batch["dones"], 1)
        obs = batch["obs"]
        actions = batch["actions"]
        next_obs = batch["next_obs"]
        goal_obs = batch["goal_obs"]

        # Qf losses
        pred_qs = [critic(obs_dict=obs, acts=actions, goal_dict=goal_obs)
                   for critic in self.nets["critic"]]

        info["critic/critic1_pred"] = pred_qs[0].mean()

        target_vf_pred = self.nets["vf"](obs_dict=next_obs, goal_dict=goal_obs).detach()
        q_target = rewards + (1. - dones) * self.algo_config.discount * target_vf_pred
        q_target = q_target.detach()

        critic_losses = []
        for (i, q_pred) in enumerate(pred_qs):
            # Calculate td error loss
            td_loss = self.td_loss_fcn(q_pred, q_target)
            info[f"critic/critic{i+1}_loss"] = td_loss
            critic_losses.append(td_loss)

        # Vf losses
        pred_qs = [critic(obs_dict=obs, acts=actions, goal_dict=goal_obs)
                        for critic in self.nets["critic_target"]]
        q_pred, _ = torch.cat(pred_qs, dim=1).min(dim=1, keepdim=True)
        q_pred = q_pred.detach()
        vf_pred = self.nets["vf"](obs)
        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.algo_config.vf_quantile + vf_sign * (1 - self.algo_config.vf_quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()
        info["vf/q_pred"] = q_pred
        info["vf/v_pred"] = vf_pred
        info["vf/v_loss"] = vf_loss

        # Return stats
        return info, critic_losses, vf_loss

    def _update_critic(self, critic_losses, vf_loss, validate=False):
        if validate:
            return

        for i, (critic_loss, critic, critic_target, optimizer) in enumerate(zip(
                critic_losses, self.nets["critic"], self.nets["critic_target"], self.optimizers["critic"]
        )):
            TorchUtils.backprop_for_loss(
                net=critic,
                optim=optimizer,
                loss=critic_loss,
                max_grad_norm=self.algo_config.critic.max_gradient_norm,
                retain_graph=False,
            )

            with torch.no_grad():
                TorchUtils.soft_update(source=critic, target=critic_target, tau=self.algo_config.target_tau)

        TorchUtils.backprop_for_loss(
            net=self.nets["vf"],
            optim=self.optimizers["vf"],
            loss=vf_loss,
            max_grad_norm=self.algo_config.critic.max_gradient_norm,
            retain_graph=False,
        )

    def _apply_adv_filter(self, adv):
        if self.algo_config.adv.clip_adv_value is not None:
            adv = adv.clamp(max=self.algo_config.adv.clip_adv_value)  # just to experiment

        filter_type = self.algo_config.adv.filter_type
        beta = self.algo_config.adv.beta
        if filter_type == "softmax":
            raise NotImplementedError
        elif filter_type == "exp":
            weights = torch.exp(adv / beta)
        elif filter_type == "binary":
            raise NotImplementedError
        else:
            raise ValueError(f"Unrecognized filter type '{filter_type}'")

        if self.algo_config.adv.use_final_clip is True:
            weights = weights.clamp(-100.0, 100.0)

        return weights[:, 0], adv

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = OrderedDict()

        log["actor/log_prob"] = info["actor/log_prob"].item()
        log["actor/loss"] = info["actor/loss"].item()

        log["critic/critic1_pred"] = info["critic/critic1_pred"].item()
        log["critic/critic1_loss"] = info["critic/critic1_loss"].item()

        log["vf/v_loss"] = info["vf/v_loss"].item()

        self._log_data_attributes(log, info, "vf/q_pred")
        self._log_data_attributes(log, info, "vf/v_pred")
        self._log_data_attributes(log, info, "adv/adv_raw")
        self._log_data_attributes(log, info, "adv/adv_score")
        self._log_data_attributes(log, info, "adv/adv_weight")

        return log

    def _log_data_attributes(self, log, info, key):
        log[key + "/max"] = info[key].max().item()
        log[key + "/min"] = info[key].min().item()
        log[key + "/mean"] = info[key].mean().item()
        log[key + "/std"] = info[key].std().item()


    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """

        # LR scheduling updates
        for lr_sc in self.lr_schedulers["critic"]:
            if lr_sc is not None:
                lr_sc.step()

        if self.lr_schedulers["actor"] is not None:
            self.lr_schedulers["actor"].step()

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

        return self.nets["actor"](obs_dict=obs_dict, goal_dict=goal_dict)
        
    def get_v_value(self, obs_dict, goal_dict=None):
        assert not self.nets.training

        return self.nets["vf"](obs_dict=obs_dict, goal_dict=goal_dict)
        
    def get_adv_value(self, obs_dict, ac, goal_dict=None):
        assert not self.nets.training

        pred_qs = [critic(obs_dict=obs_dict, acts=ac, goal_dict=goal_dict)
                        for critic in self.nets["critic_target"]]
        q_pred, _ = torch.cat(pred_qs, dim=1).min(dim=1, keepdim=True)
        vf_pred = self.nets["vf"](obs_dict, goal_dict=goal_dict)
        
        return q_pred - vf_pred

    def get_Q_value(self, obs_dict, ac, goal_dict=None):
        assert not self.nets.training

        q_value = self.nets["critic"][0](obs_dict, ac, goal_dict)

        return q_value

    def get_adv_weight(self, obs_dict, ac, goal_dict=None):
        assert not self.nets.training

        adv_value = self.get_adv_value(obs_dict=obs_dict, ac=ac, goal_dict=goal_dict)
        adv_weight = self._adv_weight_helper(adv_value)
        return adv_weight

    def _adv_weight_helper(self, adv):
        if self.algo_config.adv.clip_adv_value is not None:
            adv = adv.clamp(max=self.algo_config.adv.clip_adv_value)

        beta = self.algo_config.adv.beta
        weights = torch.exp(adv / beta)

        if self.algo_config.adv.use_final_clip is True:
            weights = weights.clamp(-100.0, 100.0)

        return weights[:, 0]
