import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import h5py
import torch

import os
import json
import h5py
import argparse
import imageio
import numpy as np
import time
import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.envs.env_base import EnvBase, EnvType

from robomimic.scripts.vis.image_utils import apply_filter

from robomimic.algo.algo import RolloutPolicy

from scipy.spatial.distance import cdist

import gc
import sys

dataset = sys.argv[1]
f = h5py.File(dataset, "r")
demos = list(f["data"].keys())
inds = np.argsort([int(elem[5:]) for elem in demos])
demos = [demos[i] for i in inds]

ckpt_path = sys.argv[2]

device = TorchUtils.get_torch_device(try_to_use_cuda=True)
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

def get_obs_at_idx(obs, i):
    d = dict()
    for key in obs:
        d[key] = obs[key][i].copy()
    for k in d:
        if "image" in k:
            d[k] = ObsUtils.process_obs(d[k], obs_modality='rgb')
    return d

embed_all = []

for demo in demos:
    print("at demo: ", demo)
    demos_obs = f["data/{}".format(demo)]["obs"]
    traj_len = len(demos_obs["agentview_image"])
    for i in range(traj_len):
        obs = get_obs_at_idx(demos_obs, i)
        obs = RolloutPolicy._prepare_observation(policy, obs)
        #for k in obs:
        #    obs[k] = torch.unsqueeze(obs[k], 1)
        embedding = policy.policy.get_obs_embedding(obs).cpu().detach().numpy()
        embed_all.append(embedding)
        for o in obs:
            obs[o].cpu().detach().numpy()
        del obs
    gc.collect()
    torch.cuda.empty_cache()

    np.save(sys.argv[3], embed_all)
