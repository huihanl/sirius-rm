import os
import h5py
import argparse
import numpy as np

from shutil import copyfile

import matplotlib.pyplot as plt

import torch
import copy
import time

"""
DEMO = -1
ROLLOUT = 0
INTV = 1
FULL_ROLLOUT = 2
"""

class TrainedPolicy:
    def __init__(self, checkpoint):
        from robomimic.utils.file_utils import policy_from_checkpoint
        self.policy = policy_from_checkpoint(ckpt_path=checkpoint)[0]
        print(self.policy.policy.nets["policy"].low_noise_eval)
        #self.policy.policy.nets["policy"].low_noise_eval = False
        self.policy.start_episode()


    def get_dist(self, obs):
        a, dist = self.policy.get_action_with_dist(obs)
        return a, dist

class EnsemblePolicy:
    def __init__(self, checkpoint_lst):
        from robomimic.utils.file_utils import policy_from_checkpoint

        self.policys = [
            policy_from_checkpoint(ckpt_path=checkpoint)[0] for checkpoint in checkpoint_lst
        ]

    def get_action(self, obs):
        obs = copy.deepcopy(obs)
        actions = []
        for policy in self.policys:
            action = policy(obs)
            actions.append(action)
        return actions

    def variance(self, obs):
        actions = self.get_action(obs)
        return np.square(np.std(np.array(actions), axis=0)).mean()

def get_obs_at_idx(obs, i):
    d = dict()
    for key in obs:
        d[key] = obs[key][i]
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
    )

    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=[]
    )

    parser.add_argument(
        "--output_dataset",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.output_dataset is not None: # make a copy in folder
        assert args.output_dataset != args.dataset

        # if not os.path.exists(args.folder):
        #     os.makedirs(args.folder)

        copyfile(args.dataset, args.output_dataset)
        out_dataset_path = os.path.expanduser(args.output_dataset)
    else:
        out_dataset_path = os.path.expanduser(args.dataset)

    f = h5py.File(out_dataset_path, "r+")

    demos = sorted(list(f["data"].keys()))

    print()
    print("Adding action modes to:", out_dataset_path, "...")

    policy = EnsemblePolicy(args.checkpoints)

    log_dir = '{}'.format(time.strftime("%y-%m-%d-%H-%M-%S"))
    os.mkdir(log_dir)

    total_variance = 0
    for ep in demos:
        print(ep)
        # store trajectory
        ep_data_grp = f["data/{}".format(ep)]
        action_modes = ep_data_grp["action_modes"][()]
        intv_labels = ep_data_grp["intv_labels"][()]
        obs = ep_data_grp["obs"]
        k = "robot0_eef_pos"

        actions = ep_data_grp["actions"][()]

        var_lst = []

        for i in range(len(obs[k])):
            o = get_obs_at_idx(obs, i)
            var = policy.variance(o)
            var_lst.append(var)

        variances = np.array(var_lst)
        variances_preintv = variances[intv_labels == -10]
        this_total = sum(variances_preintv)
        total_variance += this_total
        print("ep {}: ".format(ep), total_variance)
        
        #fig, ax = plt.subplots()
        #plt.plot(action_modes, label="action_modes")
        #plt.plot(var_lst, label="variance")
        # plt.plot(loss_lst, label="loss")

        #plt.savefig(os.path.join(log_dir, "loss_ln_{}".format(ep)))
        #plt.clf()
    print("total variance for preintv: ", total_variance)

    print("Done.")
    f.close()
