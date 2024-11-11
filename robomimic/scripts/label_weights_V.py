import os
import json
import h5py
import argparse
import imageio
import numpy as np
import math

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType

import robomimic.utils.vis_utils as VisUtils
import robomimic.utils.tensor_utils as TensorUtils

from shutil import copyfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def prepare_tensor(tensor, device=None):
    """
    Prepare raw observation dict from environment for policy.

    Args:
        ob (dict): single observation dictionary from environment (no batch dimension,
            and np.array values for each key)
    """
    tensor = TensorUtils.to_tensor(tensor)
    tensor = TensorUtils.to_batch(tensor)
    if device is not None:
        tensor = TensorUtils.to_device(tensor, device)
    tensor = TensorUtils.to_float(tensor)
    return tensor

def count_weights_with_env(
    env,
    initial_state,
    states,
    actions=None,
    algo=None,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    first=False,
):
    assert isinstance(env, EnvBase)

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    traj_len = states.shape[0]

    v_weights = []

    for i in range(traj_len):
        env.reset_to({"states" : states[i]})

        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])

        ob = env.get_observation()
        ob = prepare_tensor(ob, device=algo.device)
        ac = prepare_tensor(actions[i], device=algo.device)

        #adv_weight = algo.get_adv_weight(obs_dict=ob, ac=ac).item()
        #adv_weights.append(adv_weight)
        v_weight = algo.get_v_value(obs_dict=ob).item()
        v_weights.append(v_weight)

    return v_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="path to trained model (AWAC or IQL supported)",
    )

    parser.add_argument(
        "--output_dataset",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.output_dataset is not None: 
        assert args.output_dataset != args.dataset

        copyfile(args.dataset, args.output_dataset)
        out_dataset_path = os.path.expanduser(args.output_dataset)
    else:
        out_dataset_path = os.path.expanduser(args.dataset)

    f = h5py.File(out_dataset_path, "r+")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    algo, ckpt_dict = FileUtils.algo_from_checkpoint(ckpt_path=args.model)

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=False,
        verbose=True,
    )

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        actions = f["data/{}/actions".format(ep)][()]
        action_modes = f["data/{}/action_modes".format(ep)][()]

        weights = count_weights_with_env(
            env=env,
            initial_state=initial_state,
            states=states, 
            actions=actions,
            algo=algo,
        )
        print(weights)

        ep_data_group = f["data/{}".format(ep)]
        ep_data_group.create_dataset("weights", data=weights)

    print("Done.")
    f.close()
