import torch
import torch.nn as nn
import copy
from collections import OrderedDict

import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.value_nets as ValueNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.base_nets as BaseNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

import robomimic.utils.file_utils as FileUtils
# from robomimic.utils.file_utils import policy_from_checkpoint
from robomimic.algo import register_algo_factory_func, res_mlp_args_from_config, ValueAlgo, PolicyAlgo

@register_algo_factory_func("thrifty")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the AWAC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    return ThriftyDAgger, {}

class Ensemble:

    def __init__(self, device, checkpoints=None):
        if checkpoints is None:
            checkpoints = [
                            "/home/huihanl/expdata/spark/im/bc_xfmr/05-30-None/seed_1_ds_CoffeeSetupMug_bs_T/20240530220618/models/model_epoch_400.pth",
                            "/home/huihanl/expdata/spark/im/bc_xfmr/05-30-None/seed_2_ds_CoffeeSetupMug_bs_T/20240530220722/models/model_epoch_400.pth",
                            "/home/huihanl/expdata/spark/im/bc_xfmr/05-30-None/seed_3_ds_CoffeeSetupMug_bs_T/20240530220845/models/model_epoch_400.pth",                           
                            ] # hardcode for now

        self.n_ensembles = len(checkpoints)
        self.policies = [FileUtils.policy_from_checkpoint(ckpt_path=checkpoint)[0]
                         for checkpoint in checkpoints]
        
        self.rnn_state = [None] * self.n_ensembles
        self._rnn_horizon = 10#self.policies[0].policy.nets['policy'].rnn_horizon
        self._rnn_counter = 0

        for policy in self.policies:
            policy.start_episode()
        self.device = device

    def get_actions(self, obs, batch):
        actions = []

        with torch.no_grad():
            for i in range(self.n_ensembles):
                actions.append(self.policies[i].policy.nets['policy'].forward_train(obs_dict=obs)[0].sample())
        
        return torch.stack(actions).mean(dim=0).reshape(-1, batch['actions'].shape[-1])
    
    def get_single_step_action(self, obs):
        # obs = self._prepare_observation(obs)
        actions = []
        with torch.no_grad():
            for i in range(self.n_ensembles):
                if self.rnn_state[i] is None or self._rnn_counter % self._rnn_horizon == 0:
                    self.rnn_state[i] = self.policies[i].policy.nets['policy'].get_rnn_init_state(1, device=self.device)
                    
                act, self.rnn_state[i] = self.policies[i].policy.nets['policy'].forward_step(obs_dict=obs, rnn_state = self.rnn_state[i])
                actions.append(act)
            self._rnn_counter += 1

        return torch.stack(actions).mean(dim=0)
    
    def get_uncompressed_single_step_action(self, obs):
        actions = []
        with torch.no_grad():
            for i in range(self.n_ensembles):
                if self.rnn_state[i] is None or self._rnn_counter % self._rnn_horizon == 0:
                    self.rnn_state[i] = self.policies[i].policy.nets['policy'].get_rnn_init_state(1, device=self.device)
                    
                act, self.rnn_state[i] = self.policies[i].policy.nets['policy'].forward_step(obs_dict=obs, rnn_state = self.rnn_state[i])
                actions.append(act)
            self._rnn_counter += 1
        
        return torch.stack(actions)
    
    def _prepare_observation(self, obs):
        """
        Prepare raw observation dict from environment for policy.
        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
        """
        obs = TensorUtils.to_tensor(obs)
        obs = TensorUtils.to_batch(obs)
        obs = TensorUtils.to_device(obs, self.device)
        obs = TensorUtils.to_float(obs)
        return obs

class ThriftyDAgger(PolicyAlgo):

    def set_policy(self, checkpoints):
        self.policy = Ensemble(device=self.device, checkpoints=checkpoints)

    def _create_networks(self):
        self.nets = nn.ModuleDict()
        self.nets['critic'] = nn.ModuleList()
        self.nets['critic_target'] = nn.ModuleList()
        
        self.nets["q1"] = ValueNets.ActionValueNetwork(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            # value_bounds = (0, 1),
            goal_shapes=None,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **res_mlp_args_from_config(self.algo_config.critic.res_mlp),
        )
        self.td_loss_fcn = nn.MSELoss()

        # self.q1_target = copy.deepcopy(self.nets["q1"])
        # for p in self.q1_target.parameters():
        #     p.requires_grad = False

        # self.nets["q2"] = ValueNets.ActionValueNetwork(
        #     obs_shapes=self.obs_shapes,
        #     ac_dim=self.ac_dim,
        #     mlp_layer_dims=self.algo_config.critic.layer_dims,
        #     # value_bounds = (0, 1),
        #     goal_shapes=None,
        #     encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        #     **res_mlp_args_from_config(self.algo_config.critic.res_mlp),
        # )   

        # Policy
        # self.nets['policy'] = PolicyNets.RNNGMMActorNetwork(
        #     obs_shapes=self.obs_shapes,
        #     goal_shapes=self.goal_shapes,
        #     ac_dim=self.ac_dim,
        #     mlp_layer_dims=self.algo_config.actor_layer_dims,
        #     num_modes=self.algo_config.gmm.num_modes,
        #     min_std=self.algo_config.gmm.min_std,
        #     std_activation=self.algo_config.gmm.std_activation,
        #     low_noise_eval=self.algo_config.gmm.low_noise_eval,
        #     # stochastic_inputs=self.stochastic_inputs,
        #     encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        #     **res_mlp_args_from_config(self.algo_config.res_mlp),
        #     **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        # )

        # Load Pretrained Policy
        # self.policy = Ensemble(self.device)#policy_from_checkpoint(ckpt_path=checkpoint)[0] #

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        input_batch = dict()
        input_batch['obs'] = batch['obs']
        input_batch['next_obs'] = batch['next_obs']
        input_batch['actions'] = batch['actions']
        input_batch['rewards'] = batch['rewards']
        input_batch['dones'] = batch['dones']

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
    
    def train_on_batch(self, batch, epoch, validate=False):

        info = OrderedDict()

        # Set the correct context for this training step
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # Always run super call first
            super_info = super().train_on_batch(batch, epoch, validate=validate)
            # Train actor
            # actor_info, policy_loss = self._train_policy_on_batch(batch, epoch, validate)
            # Train critic(s)
            critic_info, critic_losses = self._train_critic_on_batch_sequence(batch, epoch, validate)
            # critic_info, critic_losses = self._train_critic_on_batch_independent(batch, epoch, validate)
            
            
            # Actor update
            # self._update_policy(policy_loss, validate)
            # Critic update
            self._update_critic(critic_losses, critic_info)

            # if epoch % 2 == 0:
            #     pass
            
            # Update info
            info.update(super_info)
            info.update(critic_info)

        # Return stats
        return info
    
    def _train_critic_on_batch_sequence(self, batch, epoch, validate=False):
        info = OrderedDict()

        batch['obs_flat'] = self.flatten_obs(batch['obs'])
        batch['next_obs_flat'] = self.flatten_obs(batch['next_obs'])
        batch['actions_flat'] = batch['actions'].reshape(-1, batch['actions'].shape[-1])
        batch['rewards_flat'] = batch['rewards'].reshape(-1, 1)
        # batch['actions_flat'] = batch['actions'][:, :-1].reshape(-1, batch['actions'].shape[-1])
        # batch['rewards_flat'] = batch['rewards'][:, :-1].reshape(-1, 1)

        q1_pred = self.q_value(self.nets['q1'], batch['obs_flat'], batch['actions_flat'], batch['rewards_flat'].shape)
        # q2_pred = self.q_value(self.nets['q2'], batch['obs_flat'], batch['actions_flat'], batch['rewards_flat'].shape)
        
        with torch.no_grad():
            # target_actions = batch['actions'][:, 1:].reshape(-1, 7) 
            target_actions = self.policy.get_actions(obs = batch['next_obs'], batch=batch)
            
            q1_target = self.q_value(self.nets['q1'], batch['next_obs_flat'], target_actions, batch['rewards_flat'].shape)
            # q2_target = self.q_value(self.nets['q2'], batch['next_obs_flat'], target_actions, batch['rewards_flat'].shape)

            backup = batch['rewards_flat'] + \
                            (1. - batch['rewards_flat']) * self.algo_config.critic.gamma * q1_target#torch.mean(torch.cat((q1_target.detach(), q2_target.detach()), dim=1), dim=1, keepdims=True)
            backup = backup.detach()

        critic1_loss = self.td_loss_fcn(q1_pred, backup)#((q1_pred - backup)**2).mean()
        # critic2_loss = nn.MSELoss()(q2_pred, backup)#((q2_pred - backup)**2).mean()

        info['critic/q1_pred'] = torch.mean(q1_pred).item()
        # info['critic/q2_pred'] = torch.mean(q2_pred).item()
        info['critic/q1_target'] = torch.mean(q1_target).item()
        # info['critic/q2_target'] = torch.mean(q2_target).item()
        info['critic/backup'] = torch.mean(backup).item()
        info['critic/critic1_loss'] = critic1_loss.item()
        # info['critic/critic2_loss'] = critic2_loss.item()
        info['critic/rewards'] = batch['rewards'].mean().item()

        critic_losses = OrderedDict(
            q1 = critic1_loss,
            # q2 = critic2_loss
        )
        return info, critic_losses

    def _train_critic_on_batch_independent(self, batch, epoch, validate=False):
        info = OrderedDict()

        batch['obs_flat'] = self.flatten_obs(batch['obs'])
        batch['next_obs_flat'] = self.flatten_obs(batch['next_obs'])
        batch['actions_flat'] = batch['actions'].reshape(-1, batch['actions'].shape[-1])

        q1_pred = self.q_value(self.nets['q1'], batch['obs_flat'], batch['actions_flat'], batch['rewards'])[:, -1]
        q2_pred = self.q_value(self.nets['q2'], batch['obs_flat'], batch['actions_flat'], batch['rewards'])[:, -1]
        
        with torch.no_grad():
            target_actions = self.policy.get_actions(obs = batch['next_obs'], batch=batch)
            q1_target = self.q_value(self.nets['q1'], batch['next_obs_flat'], target_actions, batch['rewards'])[:, -1]
            q2_target = self.q_value(self.nets['q2'], batch['next_obs_flat'], target_actions, batch['rewards'])[:, -1]

            # need to pad the batch
            backup = batch['rewards'][:, -1] + \
                            (1 - batch['rewards'][:, -1]) * self.algo_config.critic.gamma * torch.min(q1_target.detach(), q2_target.detach())
            
            # print(q1_pred.shape, q1_target.shape, backup.shape)
            # if torch.sum(batch['rewards'][:, -1]) > 0:
            #     print(backup)

        critic1_loss = nn.MSELoss()(q1_pred, backup)#((q1_pred - backup)**2).mean()
        critic2_loss = nn.MSELoss()(q2_pred, backup)#((q2_pred - backup)**2).mean()

        info['critic/q1_pred'] = torch.mean(q1_pred).item()
        info['critic/q2_pred'] = torch.mean(q2_pred).item()
        info['critic/q1_target'] = torch.mean(q1_target).item()
        info['critic/q2_target'] = torch.mean(q2_target).item()
        info['critic/backup'] = torch.mean(backup).item()
        info['critic/critic1_loss'] = critic1_loss.item()
        info['critic/critic2_loss'] = critic2_loss.item()
        info['critic/rewards'] = batch['rewards'][:, -1].mean().item()

        critic_losses = OrderedDict(
            q1 = critic1_loss,
            q2 = critic2_loss
        )
        return info, critic_losses
    
    def q_value(self, q_net, obs, actions, rewards_shape):
        q_pred = q_net.forward(obs, actions)[..., 0].reshape(rewards_shape)
        return q_pred

    def flatten_obs(self, obs):
        obs_flat = OrderedDict()
        for k in obs.keys():
            if len(obs[k].shape) == 5:
                obs_flat[k] = obs[k].reshape(-1, *obs[k].shape[-3:])
                # obs_flat[k] = obs[k][:, :-1].reshape(-1, *obs[k].shape[-3:])
            else:
                obs_flat[k] = obs[k].reshape(-1, obs[k].shape[-1])
                # obs_flat[k] = obs[k][:, :-1].reshape(-1, obs[k].shape[-1])
        return obs_flat

    def _update_critic(self, critic_losses, critic_info):
        info = OrderedDict()
        for k in critic_losses.keys():
            policy_grad_norms = TorchUtils.backprop_for_loss(
                net=self.nets[k],
                optim=self.optimizers[k],
                loss=critic_losses[k],
                max_grad_norm=self.algo_config.max_gradient_norm,
                retain_graph=False,
            )
            info[f"policy_grad_norms_{k}"] = policy_grad_norms
        
        return info

    def log_info(self, info):
        log = OrderedDict()
        
        for k, v in info.items():
            log[k] = v

        return log

    def get_q_safety(self, obs, actions):
        q1 = self.nets['q1'].forward(obs, actions)[..., 0]
        return float(q1.cpu().detach().numpy())
        # q2 = self.nets['q2'].forward(obs, actions)[..., 0]
        # return float(torch.min(q1, q2).cpu().detach().numpy())

    def get_single_step_action(self, obs):
        return self.policy.get_single_step_action(obs)

    def get_uncompressed_single_step_action(self, obs):
        return self.policy.get_uncompressed_single_step_action(obs)

if __name__ == "__main__":
    Ensemble()
