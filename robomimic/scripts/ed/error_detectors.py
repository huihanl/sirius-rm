import os
import h5py
import argparse
import numpy as np

from shutil import copyfile

import matplotlib.pyplot as plt

import torch
import copy
import time

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.utils.file_utils import policy_from_checkpoint

import faiss

from sklearn import svm

from robomimic.algo.algo import RolloutPolicy

from collect_hitl_demos_ed import load_detector_config

def process_image(img, transpose=True):
    image = torch.from_numpy(img)[None, :, :, :, :].cuda().float()
    if transpose:
        image = image.permute(0, 1, 4, 2, 3) / 255.
    return image

def process_action(action):
    return torch.from_numpy(action)[None, :].cuda().float()

def get_obs_at_idx(obs, i):
    d = dict()
    for key in obs:
        d[key] = obs[key][i]
    return d

def process_shadowing_mode(obs):
    for key in obs:
        if "image" in key:
            obs[key] = ObsUtils.process_obs(obs[key], obs_modality='rgb')
    return obs

class ErrorDetector:
    def __init__(self):
        # shadowing: if human shadowing using recorded obs, set to True
        self.shadowing_node = False
        pass

    def evaluate(self, obs):
        assert NotImplementedError

    def evaluate_trajectory(self, obs_np):
        assert NotImplementedError

    def above_threshold(self, value):
        assert NotImplementedError
    
    def reset(self):
        pass

class VAEMoMaRT(ErrorDetector):
    def __init__(self, checkpoint, threshold):
        super(VAEMoMaRT, self).__init__()
        self.rollout_policy = policy_from_checkpoint(ckpt_path=checkpoint)[0]
        self.policy = self.rollout_policy.policy.nets["policy"]
        self.threshold = threshold
        self.seq_length = 10
        #self.image_key = "agentview_image"

        ckpt_dict = policy_from_checkpoint(ckpt_path=checkpoint)[1]
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        self.rollout_horizon = config.experiment.rollout.horizon

        self._value_history = []

    def evaluate(self, obs):
        vae_outputs = self.policy.forward(obs)
        return vae_outputs

    def evaluate_trajectory(self, obs_np):
        key_choices = list(obs_np.keys())
        key = "birdview_image" if "birdview_image" in key_choices else "agentview_image"
        recons_loss_lst = []
        kl_loss_lst = []
        reconstructions_lst = []
        for i in range(len(obs_np[key]) - self.seq_length):
            obs_input = {}
            obs_input[key] = obs_np[key][i : i + self.seq_length]
            obs_input[key] = process_image(obs_input[key])
            vae_outputs = self.evaluate(obs_input)

            reconstructions = vae_outputs["reconstructions"][key]
            recons_loss = vae_outputs["reconstruction_loss"].item()
            kl_loss = vae_outputs["kl_loss"].item()

            reconstructions_lst.append(reconstructions)
            recons_loss_lst.append(recons_loss)
            kl_loss_lst.append(kl_loss)

        # clear anomaly data
        for i in range(10):
            recons_loss_lst[i] = min(recons_loss_lst[i], 0.01)
            kl_loss_lst[i] = min(kl_loss_lst[i], 0.01)

        # append 0 for initial data (no history)
        recons_loss_lst = [0] * 10 + recons_loss_lst
        kl_loss_lst = [0] * 10 + kl_loss_lst

        return recons_loss_lst

    def above_threshold(self, value):
        return value >= self.threshold

    def metric(self):
        return "Reconstruction Loss"

    def horizon(self):
        return self.rollout_horizon

    def human_intervene(self, obs_buffer): # of the current last observation
        if len(obs_buffer) < self.seq_length: # not enough history yet, always good
            return False
        else:
            key_choices = list(obs_buffer[0].keys())
            obs_input = {}
            for key in ["agentview_image", "robot0_eye_in_hand_image"]:
                obs_np = []
                for obs in obs_buffer[-self.seq_length:]:
                    obs_np.append(obs[key])
                obs_np = np.array(obs_np).copy()

                obs_input[key] = obs_np           

            if self.shadowing_node:
                obs_input = process_shadowing_mode(obs_input)

            obs_input = RolloutPolicy._prepare_observation(self.rollout_policy, obs_input)
            vae_outputs = self.evaluate(obs_input)
            recons_loss = vae_outputs["reconstruction_loss"].item()
            print(recons_loss)

            self._value_history.append(recons_loss)

            return self.above_threshold(recons_loss) # if greater, is error

    def reset(self):
        self._value_history = []

class VAE_Action(ErrorDetector):
    def __init__(self, checkpoint, threshold):
        self.policy = policy_from_checkpoint(ckpt_path=checkpoint)[0].policy.nets["policy"]
        self.threshold = threshold
        self.seq_length = 10
        self.image_key = "agentview_image"

        ckpt_dict = policy_from_checkpoint(ckpt_path=checkpoint)[1]
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        self.rollout_horizon = config.experiment.rollout.horizon

    def evaluate(self, action, obs):
        vae_outputs = self.policy.forward(action, obs)
        return vae_outputs

    def evaluate_trajectory(self, action_np, obs_np):
        key = self.image_key
        recons_loss_lst = []
        kl_loss_lst = []
        reconstructions_lst = []
        for i in range(len(obs_np[key]) - self.seq_length):
            obs_input = {}
            obs_input[key] = obs_np[key][i : i + self.seq_length]
            obs_input[key] = process_image(obs_input[key])
            action = action_np[i : i + self.seq_length]
            action = process_action(action)
            vae_outputs = self.evaluate(action, obs_input)

            recons_loss = vae_outputs["reconstruction_loss"].item()
            kl_loss = vae_outputs["kl_loss"].item()

            recons_loss_lst.append(recons_loss)
            kl_loss_lst.append(kl_loss)

        # clear anomaly data
        for i in range(10):
            recons_loss_lst[i] = min(recons_loss_lst[i], 0.01)
            kl_loss_lst[i] = min(kl_loss_lst[i], 0.01)

        # append 0 for initial data (no history)
        recons_loss_lst = [0] * 10 + recons_loss_lst
        kl_loss_lst = [0] * 10 + kl_loss_lst

        return recons_loss_lst

    def above_threshold(self, value):
        return value >= self.threshold

    def metric(self):
        return "Reconstruction Loss"

    def horizon(self):
        return self.rollout_horizon

class VAEGoal(VAEMoMaRT):
    def __init__(self, checkpoint, threshold):
        super(VAEGoal, self).__init__(checkpoint, threshold)
        self.sampling_num = 1024 # hardcode for now
        self.seq_length = 1 # only use the current observation

        self._value_history = []

    def evaluate(self, obs):
        key_choices = ["agentview_image", "robot0_eye_in_hand_image"] #list(obs.keys())
        for key in key_choices:
            obs[key] = torch.squeeze(obs[key], 0)
            obs[key] = obs[key].expand(self.sampling_num, -1, -1, -1)
        
        recons = self.policy.decode(obs_dict=obs, n=self.sampling_num)
        
        recons_all = []
        for key in key_choices:
            recons_flat = torch.flatten(recons[key], start_dim=1)
            recons_all.append(recons_flat)
        recons_flat = torch.cat(recons_all, dim=1)

        variance = torch.var(recons_flat, dim=0).mean()
        return variance.item()

    def evaluate_trajectory(self, obs_np):
        key_choices = list(obs_np.keys())
        
        recons_loss_lst = []
        kl_loss_lst = []
        reconstructions_lst = []
        for i in range(len(obs_np[key]) - self.seq_length):
            obs_input = {}
            obs_input[key] = obs_np[key][i : i + self.seq_length]
            obs_input[key] = process_image(obs_input[key])
            vae_outputs = self.evaluate(obs_input)

            reconstructions = vae_outputs["reconstructions"][key]
            recons_loss = vae_outputs["reconstruction_loss"].item()
            kl_loss = vae_outputs["kl_loss"].item()

            reconstructions_lst.append(reconstructions)
            recons_loss_lst.append(recons_loss)
            kl_loss_lst.append(kl_loss)

        # clear anomaly data
        for i in range(10):
            recons_loss_lst[i] = min(recons_loss_lst[i], 0.01)
            kl_loss_lst[i] = min(kl_loss_lst[i], 0.01)

        # append 0 for initial data (no history)
        recons_loss_lst = [0] * 10 + recons_loss_lst
        kl_loss_lst = [0] * 10 + kl_loss_lst

        return recons_loss_lst

    def above_threshold(self, value):
        return value >= self.threshold

    def metric(self):
        return "VAE Goal Variance: "

    def horizon(self):
        return self.rollout_horizon

    def human_intervene(self, obs_buffer): # of the current last observation
        if len(obs_buffer) < self.seq_length: # not enough history yet, always good
            return False
        else:
            obs = obs_buffer[-1].copy()

            if self.shadowing_node:
                obs = process_shadowing_mode(obs)

            obs_input = RolloutPolicy._prepare_observation(self.rollout_policy, obs)
            variance = self.evaluate(obs_input)

            self._value_history.append(variance)

            print(self.metric(), variance)
            return self.above_threshold(variance) # if greater, is error
        
    def reset(self):

        self._value_history = []

        pass
    
class Ensemble(ErrorDetector):
    def __init__(self, checkpoints, threshold):
        super(Ensemble, self).__init__()
        self.policies = [policy_from_checkpoint(ckpt_path=checkpoint)[0]
                         for checkpoint in checkpoints]

        for policy in self.policies:
            policy.start_episode(eval_policy_only=False)

        self.threshold = threshold

        ckpt_dict = policy_from_checkpoint(ckpt_path=checkpoints[0])[1]
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        self.rollout_horizon = config.experiment.rollout.horizon
        self.no_gripper_action = False
        self.seq_length = 10

        self._value_history = []

    def evaluate(self, obs, no_gripper_action=False):
        # for k in obs:
        #     if "image" in k:
        #         obs[k] = ObsUtils.process_obs(obs[k], obs_modality='rgb')

        actions = []
        for policy in self.policies:
            action = policy(obs)
            actions.append(action)
        action_variance = np.square(np.std(np.array(actions), axis=0)).mean()
        if no_gripper_action:
            for i in range(len(actions)):
                actions[i] = actions[i][:6]
            action_variance = np.square(np.std(np.array(actions), axis=0)).mean()
        return action_variance

    def evaluate_trajectory(self, obs_np):
        key = list(obs_np.keys())[0] # any random key
        var_lst = []
        for i in range(len(obs_np[key])):
            obs_input = get_obs_at_idx(obs_np, i)
            var_output = self.evaluate(obs_input, self.no_gripper_action)
            var_lst.append(var_output)
        return var_lst

    def above_threshold(self, value):
        return value >= self.threshold

    def metric(self):
        return "Action Variance: "

    def horizon(self):
        return self.rollout_horizon

    def human_intervene(self, obs_buffer): # of the current last observation
        if len(obs_buffer) < self.seq_length: # not enough history yet, always good
            return False
        else:
            obs_np = obs_buffer[-1].copy()
            
            if self.shadowing_node:
                obs_np = process_shadowing_mode(obs_np)

            # Deal with ToolHang potentially
            obs_np.pop('frame_is_assembled', None)
            obs_np.pop('tool_on_frame', None)

            var_output = self.evaluate(obs_np, self.no_gripper_action)

            self._value_history.append(var_output)

            print(self.metric(), var_output)
            return self.above_threshold(var_output) # if greater, is error
        
    def reset(self):
        for policy in self.policies:
            policy.start_episode(eval_policy_only=False)

        self._value_history = []
        
class BCDreamer_OOD(ErrorDetector):
    def __init__(self, checkpoint, threshold, demos_embedding_path, eval_method, num_future, use_imagine=True):
        
        super(BCDreamer_OOD, self).__init__()
        
        self.rollout_policy = policy_from_checkpoint(ckpt_path=checkpoint)[0] #.nets["policy"]
        self.policy = self.rollout_policy.policy
        self.threshold = threshold
        self.seq_length = 10
        self.eval_method = eval_method
        assert eval_method in ["mean", "first", "last"]
        self.num_future = num_future

        ckpt_dict = policy_from_checkpoint(ckpt_path=checkpoint)[1]
        self.config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

        # Load embeddings and latent_lst
        self.demos_embedding = np.load(demos_embedding_path, 
                                       allow_pickle=True)
        for i in range(2):
            self.demos_embedding = np.squeeze(self.demos_embedding, 1)

        self._img_embedding_history = []
        self._value_history = []

        # for offline rollouts
        self.use_imagine = use_imagine

    # For online rollouts in HITL loop
    def evaluate(self, obs):
        
        #self.rollout_policy(obs)
        obs = RolloutPolicy._prepare_observation(self.rollout_policy, obs)
        imagined_embeddings = self.policy.imagine(obs)
        nearest_neighbor_distances = self._compute_nearest_neighbor_distance(imagined_embeddings, 
                                                                        self.demos_embedding)

        if self.eval_method == "first":
            nearest_neighbor_distances = nearest_neighbor_distances[:,0].mean()
        elif self.eval_method == "last":
            nearest_neighbor_distances = nearest_neighbor_distances[:,-1].mean()
        elif self.eval_method == "mean":
            nearest_neighbor_distances = np.mean(nearest_neighbor_distances)
        print(nearest_neighbor_distances)
        return nearest_neighbor_distances

    def human_intervene(self, obs_buffer): # of the current last observation
            
        
        # to hardcode for obs_buffer in hitl script
        if type(obs_buffer) is list:
            obs = obs_buffer[-1]
        
        if self.shadowing_node:
            obs = process_shadowing_mode(obs)
        
        nearest_neighbor_distance = self.evaluate(obs)
        
        self._value_history.append(nearest_neighbor_distance)
        
        if type(obs_buffer) is list and len(obs_buffer) < 12:
            return False 
        
        return self.above_threshold(nearest_neighbor_distance)

    # For visualization of offline rollouts
    def evaluate_trajectory(self, latent_lst, use_img_eval=False):
        # Takes in latent_lst for now, rather than raw obs
        # Compute nearest neighbor distances
        nearest_neighbor_distances = self._compute_nearest_neighbor_distance(latent_lst, 
                                                                             self.demos_embedding)
        if self.use_imagine or use_img_eval:
            # set first 13 to 0
            nearest_neighbor_distances[:13] = 0
            
        return nearest_neighbor_distances # error states are larger values

    def above_threshold(self, value):
        return value >= self.threshold  

    def _query_nn(self, query_vectors, dataset, k):
        dimension = dataset.shape[-1]
        dataset /= np.linalg.norm(dataset, axis=1)[:, np.newaxis]
        query_vectors /= np.linalg.norm(query_vectors, axis=1)[:, np.newaxis]

        index = faiss.IndexFlatL2(dimension)
        index.add(dataset)

        distances, indices = index.search(query_vectors, k)

        ave_dist_lst = []
        for i, (query_dist, query_idx) in enumerate(zip(distances, indices)):
            total_dist = 0
            for rank, (dist, idx) in enumerate(zip(query_dist, query_idx), start=1):
                total_dist += dist
            ave_dist_lst.append(total_dist / len(query_dist))

        return ave_dist_lst
    
    def _moving_average_past(self, data, window_size):
        assert window_size > 0, "Window size must be greater than 0"
        
        smoothed_data = np.zeros(len(data))
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            end = i + 1
            smoothed_data[i] = np.mean(data[start:end])
        
        return smoothed_data

    def _compute_nearest_neighbor_distance(self, latent_lst, demos_embedding):
        A,B,C = latent_lst.shape
        latent_lst = np.reshape(latent_lst, (A * B, C))
        ave_dist_lst = self._query_nn(latent_lst, demos_embedding, k=5)
        ave_dist_lst = self._moving_average_past(ave_dist_lst, window_size=3)

        ave_dist_lst = np.reshape(ave_dist_lst, (A, B))
        return ave_dist_lst
    
    def metric(self):
        # for plotting figure title
        if self.use_imagine:
            return "OOD with Imagined Embedding"
        else:
            return "OOD with Current Embedding"
    
    def horizon(self):
        return 400
    
    def reset(self):
        self.rollout_policy.start_episode(eval_policy_only=False)
        self._value_history = []

class BCDreamer_Failure(ErrorDetector):
    def __init__(self, 
                 checkpoint, 
                 threshold, 
                 threhold_history, 
                 threshold_count,
                 eval_method, 
                 eval_idx,
                 use_prob=False):
        
        super(BCDreamer_Failure, self).__init__()
        
        self.rollout_policy = policy_from_checkpoint(ckpt_path=checkpoint)[0] #.nets["policy"]
        self.policy = self.rollout_policy.policy
        self.threshold = threshold
        self.threhold_history = threhold_history
        self.threshold_count = threshold_count
        self.eval_method = eval_method
        self.eval_idx = eval_idx

        #ckpt_dict = policy_from_checkpoint(ckpt_path=checkpoint)[1]
        #self.config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

        self.use_prob = use_prob

        self._value_history = {"reward_label": [], "reward_prob": []}

    def evaluate(self, obs):
        obs = copy.deepcopy(obs)
        
        #self.rollout_policy(obs)
        
        obs = RolloutPolicy._prepare_observation(self.rollout_policy, obs)
        # print(obs["agentview_image"].shape)
        imagined_embeddings = self.policy.imagine(obs)
        reward_label, reward_prob = self.policy.compute_reward_test_time(imagine="all")

        # (num_future * horizon)
        reward_label = np.int8((reward_label == 1))
        reward_prob = reward_prob[:,:,1]

        if self.eval_method == "idx":
            # take the given index from eval_method
            reward_label = reward_label[:,self.eval_idx] 
            reward_prob = reward_prob[:,self.eval_idx]
        else:
            # take the mean
            assert self.eval_method == "mean"
            
        # average across multiple futures
        reward_label = reward_label.mean()
        reward_prob = reward_prob.mean()

        self._value_history["reward_label"].append(reward_label)
        self._value_history["reward_prob"].append(reward_prob)

        return reward_label, reward_prob

    def evaluate_trajectory(self, s, use_prob_eval=False):
        if self.use_prob or use_prob_eval:
            s[:12] = 0.
        return s

    def above_threshold(self, value, use_prob_eval=False):
        raise NotImplementedError("Not used here")
        
    def metric(self):
        # for plotting figure title
        return "Failure Probablity"
    
    def horizon(self):
        return 400

    def human_intervene(self, obs): # of the current last observation
        
        if type(obs) is list and len(obs) < 10:
            self._value_history["reward_label"].append(0)
            self._value_history["reward_prob"].append(1)
            return False # no intervention
        
        # to hardcode for obs_buffer in hitl script
        if type(obs) is list:
            obs = obs[-1]
        
        if self.shadowing_node:
            obs = process_shadowing_mode(obs)
        
        _, _ = self.evaluate(obs)
        # needs to be true for last 10 steps
        
        considered_segment = np.array(self._value_history["reward_label"][-self.threhold_history:])
        if sum(considered_segment > self.threshold) > self.threshold_count:
            print("intervene")
            return True
        else:
            return False
        
    def reset(self):
        self.rollout_policy.start_episode(eval_policy_only=False)
        self._value_history = {"reward_label": [], "reward_prob": []}
        
class BCDreamer_SVM(ErrorDetector):
    def __init__(self, 
                 checkpoint, 
                 threshold, 
                 demos_embedding_path, 
                 svm_config={
                     "nu": 0.1,
                     "kernel": "rbf",
                 },
                 use_prob=False):
        
        super(BCDreamer_SVM, self).__init__()
        
        self.rollout_policy = policy_from_checkpoint(ckpt_path=checkpoint)[0] #.nets["policy"]
        self.policy = self.rollout_policy.policy
        
        self.threshold = threshold
    
        # Load embeddings and latent_lst
        self.demos_embedding = np.load(demos_embedding_path, 
                                       allow_pickle=True)
        for i in range(2):
            self.demos_embedding = np.squeeze(self.demos_embedding, 1)
            
        self.svm_classifier = svm.OneClassSVM(nu=svm_config["nu"], 
                                              kernel=svm_config["kernel"], 
                                              gamma=0.1)
        
        self.svm_classifier.fit(self.demos_embedding)
        
        self._value_history = []

    def evaluate(self, obs):

        obs = copy.deepcopy(obs)
        
        #self.rollout_policy(obs)
        
        obs = RolloutPolicy._prepare_observation(self.rollout_policy, obs)
        # print(obs["agentview_image"].shape)
        imagined_embeddings = self.policy.imagine(obs)
        
        # if the imagined embeddings are all zero
        if imagined_embeddings.sum() == 0:
            return 100 # return a positive value
        
        score = self.svm_classifier.decision_function(imagined_embeddings[:,-1,:])
        print(score[0])
        print("Score metrics: max {}, min {}, mean {}, std {}".format(score.max(), score.min(), score.mean(), score.std()))
        return score.mean()  

    def evaluate_trajectory(self, latent_lst, use_img_eval=False):

        similarities = self.svm_classifier.decision_function(latent_lst)
        
        if use_img_eval:
            # set first 13 to dummy values
            similarities[:13] = similarities[13]
        
        return similarities
    
    def above_threshold(self, value):
        return value <= self.threshold
    
    def metric(self):
        # for plotting figure title
        return "SVM Classifier"
    
    def horizon(self):
        return 400

    def human_intervene(self, obs): # of the current last observation
        
        if type(obs) is list and len(obs) < 13:
            return False
        
        # to hardcode for obs_buffer in hitl script
        if type(obs) is list:
            obs = obs[-1]
        
        if self.shadowing_node:
            obs = process_shadowing_mode(obs)
        
        score = self.evaluate(obs)
        print(score)
        
        self._value_history.append(score)
        
        return self.above_threshold(score)

    def reset(self):
        self.rollout_policy.start_episode(eval_policy_only=False)
        self._value_history = []
        
    def reset_history_only(self):
        self.policy.reset_history_only()
        self._value_history = []

class PATO(ErrorDetector):
    def __init__(self, checkpoints, vae_goal_th, ensemble_th):
        
        super(PATO, self).__init__()
        
        self.vae_goal_detector = VAEGoal(checkpoints[0], vae_goal_th)
        self.vae_goal_detector.shadowing_node = True # hardcode now

        assert len(checkpoints[1:]) == 5, "Need 5 checkpoints for ensemble"
        for checkpoint in checkpoints[1:]:
            assert "bs_sampling_T_" in checkpoint, "Need bootstrap checkpoint"
        
        self.ens_detector = Ensemble(checkpoints[1:], ensemble_th) 
        self.ens_detector.shadowing_node = True # hardcode now

        self._value_history = {"vae_intv": [], "ens_intv": []}
        
    def human_intervene(self, obs_buffer):
        vae_intv = self.vae_goal_detector.human_intervene(obs_buffer)
        ens_intv = self.ens_detector.human_intervene(obs_buffer)

        self._value_history["vae_intv"] = self.vae_goal_detector._value_history
        self._value_history["ens_intv"] = self.ens_detector._value_history

        if vae_intv:
            print("VAE Goal intervene")
            return True

        if ens_intv:
            print("Ensemble intervene")
            return True
        
        return False
    
    def reset(self):
        self.vae_goal_detector.reset()
        self.ens_detector.reset()

        self._value_history = {"vae_intv": [], "ens_intv": []}


class BCDreamer_Combined(ErrorDetector):
    def __init__(self, checkpoints, 
                 ood_type, 
                 num_future, 
                 demos_embedding_path,
                 ):
        
        super(BCDreamer_Combined, self).__init__()
        
        self.rollout_policy = policy_from_checkpoint(ckpt_path=checkpoints[0])[0] #.nets["policy"]
        self.policy = self.rollout_policy.policy
        self.ood_type = ood_type
        self.num_future = num_future
        
        detector_config = load_detector_config("bc_dreamer_failure")
        self._init_classifier_params(**detector_config)
        
        self._init_demo_embedding(demos_embedding_path)
        if ood_type == "svm":
            detector_config = load_detector_config("bc_dreamer_svm")
            self._init_svm_params(**detector_config)
        elif ood_type == "ood":
            detector_config = load_detector_config("bc_dreamer_ood")
            self._init_neighbor_params(**detector_config)
            
    def _init_demo_embedding(self, demos_embedding_path):
        self.demos_embedding = np.load(demos_embedding_path, 
                                       allow_pickle=True)
        for i in range(2):
            self.demos_embedding = np.squeeze(self.demos_embedding, 1)
            
    def _init_classifier_params(self, threshold, eval_method, eval_idx, threhold_history, threshold_count):
        self.rew_threshold = threshold
        self.rew_eval_method = eval_method
        self.rew_eval_idx = eval_idx
        self.rew_threhold_history = threhold_history
        self.rew_threshold_count = threshold_count
        self.rew_value_history = {"reward_label": [], "reward_prob": []}

    def _init_svm_params(self, threshold, demos_embedding_path, svm_config):
        self.svm_threshold = threshold
            
        self.svm_classifier = svm.OneClassSVM(nu=svm_config["nu"], 
                                              kernel=svm_config["kernel"], 
                                              gamma=0.1)
        self.svm_classifier.fit(self.demos_embedding)
        
    def _init_neighbor_params(self, threshold, demos_embedding_path, eval_method, num_future):

        self.nn_threshold = threshold
        self.nn_seq_length = 10
        self.eval_method = eval_method
        assert eval_method in ["mean", "first", "last"]

        self._img_embedding_history = []
        self._nn_value_history = []
        
    def _ood_human_intervene(self, imagined_embeddings):
        if self.ood_type == "svm":
            return self._svm_evaluate(imagined_embeddings)
        elif self.ood_type == "ood":
            return self._neighbor_evaluate(imagined_embeddings)
 
    def _svm_evaluate(self, imagined_embeddings):
        #return False
        
        # if the imagined embeddings are all zero
        if imagined_embeddings.sum() == 0:
            return False # do not intervene
        #print(imagined_embeddings.shape)
        score = self.svm_classifier.decision_function(imagined_embeddings[:,-1,:])
        print(score.mean())
        print("Score metrics: max {}, min {}, mean {}, std {}".format(score.max(), score.min(), score.mean(), score.std()))
        if score.mean() < self.svm_threshold:
            print("SVM intervene")
            return True
        else:
            return False
        
    def _neighbor_evaluate(self, imagined_embeddings):

        nearest_neighbor_distances = self._compute_nearest_neighbor_distance(
                                        imagined_embeddings, 
                                        self.demos_embedding)

        if self.eval_method == "first":
            nearest_neighbor_distances = nearest_neighbor_distances[:,0].mean()
        elif self.eval_method == "last":
            nearest_neighbor_distances = nearest_neighbor_distances[:,-1].mean()
        elif self.eval_method == "mean":
            nearest_neighbor_distances = np.mean(nearest_neighbor_distances)

        if len(self._nn_value_history) < 12:
            nearest_neighbor_distances = 0
        
        self._nn_value_history.append(nearest_neighbor_distances)

        print(nearest_neighbor_distances)

        return nearest_neighbor_distances >= self.nn_threshold
 
    def _query_nn(self, query_vectors, dataset, k):
        dimension = dataset.shape[-1]
        dataset /= np.linalg.norm(dataset, axis=1)[:, np.newaxis]
        query_vectors /= np.linalg.norm(query_vectors, axis=1)[:, np.newaxis]

        index = faiss.IndexFlatL2(dimension)
        index.add(dataset)

        distances, indices = index.search(query_vectors, k)

        ave_dist_lst = []
        for i, (query_dist, query_idx) in enumerate(zip(distances, indices)):
            total_dist = 0
            for rank, (dist, idx) in enumerate(zip(query_dist, query_idx), start=1):
                total_dist += dist
            ave_dist_lst.append(total_dist / len(query_dist))

        return ave_dist_lst
    
    def _moving_average_past(self, data, window_size):
        assert window_size > 0, "Window size must be greater than 0"
        
        smoothed_data = np.zeros(len(data))
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            end = i + 1
            smoothed_data[i] = np.mean(data[start:end])
        
        return smoothed_data

    def _compute_nearest_neighbor_distance(self, latent_lst, demos_embedding):
        A,B,C = latent_lst.shape
        latent_lst = np.reshape(latent_lst, (A * B, C))
        ave_dist_lst = self._query_nn(latent_lst, demos_embedding, k=5)
        ave_dist_lst = self._moving_average_past(ave_dist_lst, window_size=3)

        ave_dist_lst = np.reshape(ave_dist_lst, (A, B))
        return ave_dist_lst

    def _failure_human_intervene(self, imagined_embeddings):      
            
        reward_label, reward_prob = self.policy.compute_reward_test_time(imagine="all")

        # (num_future * horizon)
        reward_label = np.int8((reward_label == 1))
        reward_prob = reward_prob[:,:,1]

        if self.rew_eval_method == "idx":
            # take the given index from eval_method
            reward_label = reward_label[:,self.rew_eval_idx] 
            reward_prob = reward_prob[:,self.rew_eval_idx]
        else:
            # take the mean
            assert self.rew_eval_method == "mean"
            
        # average across multiple futures
        reward_label = reward_label.mean()
        reward_prob = reward_prob.mean()

        self.rew_value_history["reward_label"].append(reward_label)
        self.rew_value_history["reward_prob"].append(reward_prob)

        print(self.rew_value_history["reward_label"][-self.rew_threhold_history:])

        considered_segment = np.array(self.rew_value_history["reward_label"][-self.rew_threhold_history:])
        if sum(considered_segment > self.rew_threshold) > self.rew_threshold_count:
            #print("intervene")
            return True
        else:
            return False
        

    def human_intervene(self, obs):
        # not enough history
        if type(obs) is list and len(obs) < 13:
            self.rew_value_history["reward_label"].append(0)
            self.rew_value_history["reward_prob"].append(1)
            return False
        
        if type(obs) is list:
            obs = obs[-1]
            
        obs = copy.deepcopy(obs)
        
        if self.shadowing_node:
            obs = process_shadowing_mode(obs)
        
        obs = RolloutPolicy._prepare_observation(self.rollout_policy, obs)
        imagined_embeddings = self.policy.imagine(obs)
        
        ood_intervene = self._ood_human_intervene(imagined_embeddings)
        
        failure_intervene = self._failure_human_intervene(imagined_embeddings)
        
        if ood_intervene:
            print()
            print('\033[96m OOD intervene \033[0m')
            print()
        if failure_intervene:
            print()            
            print('\033[96m Failure intervene \033[0m')
            print()
        return ood_intervene or failure_intervene
    
    def reset(self):
        self.rollout_policy.start_episode(eval_policy_only=False)
        
    def reset_history_only(self):
        self.policy.reset_history_only()



class ThriftyDAggerED(ErrorDetector):
    def __init__(self, checkpoints, q_th, ensemble_th):

        super(ThriftyDAggerED, self).__init__()

        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=checkpoints[0])
        self.policy = policy.policy
        self.q_th = q_th
        self.ensemble_th = ensemble_th

        assert len(checkpoints) == 6 # 1 + 5

        self.policy.set_policy(checkpoints=checkpoints[1:])
        
        self.actions = None
        self.action_already_generated = False
        
        self._value_history = {"ens_intv": [], "q_intv": []}

    def human_intervene(self, obs_buffer):
        ob = obs_buffer[-1]
       
        if self.shadowing_node:
            ob = process_shadowing_mode(ob) 

        #self.get_action(copy.deepcopy(ob))
 
        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.policy.device)
        ob = TensorUtils.to_float(ob)
        
        assert self.action_already_generated, "Call get_action before human_intervene"
        self.action_already_generated = False
        action = np.mean(self.actions, axis=0)
        
        action = TensorUtils.to_tensor(action)
        action = TensorUtils.to_device(action, self.policy.device)
        action = TensorUtils.to_float(action)
        
        ensemble_variance = np.mean(np.square(np.std(self.actions, axis=0)))
        q_val = self.policy.get_q_safety(ob, action)

        self._value_history["ens_intv"].append(ensemble_variance)
        self._value_history["q_intv"].append(q_val)

        print("\033[93m Q value: {} \033[0m".format(q_val))

        if ensemble_variance > self.ensemble_th:
            print("Ensemble intervene")
            return True
        if q_val < self.q_th:
            print("Q val intervene")
            return True
        return False

    def get_action(self, ob):

        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.policy.device)
        ob = TensorUtils.to_float(ob)
        
        self.actions = self.policy.get_uncompressed_single_step_action(ob).cpu().detach().numpy()
        self.action_already_generated = True
        
        return np.mean(self.actions, axis=0)[0]

    def reset(self):

        ens_policies = self.policy.policy.policies
        for policy in ens_policies:
            policy.start_episode()

        self._value_history = {"ens_intv": [], "q_intv": []}
