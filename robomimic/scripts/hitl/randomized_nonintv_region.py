import os
import h5py
import argparse
import numpy as np

from shutil import copyfile


DEMO = -1
ROLLOUT = 0
INTV = 1
FULL_ROLLOUT = 2

PRE_INTV = -10

import random

def _check_grasp_success(env):
    grasped = env.env._check_grasp(
        gripper=env.env.robots[0].gripper,
        object_geoms=[g for g in env.env.nuts[0].contact_geoms])
    return grasped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--pre_intv_len",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--filter_key",
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

    filter_key = args.filter_key
    if filter_key is not None:
        print("using filter key: {}".format(filter_key))
        demos = sorted([elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])])

    for ep in demos:
        # store trajectory
        ep_data_grp = f["data/{}".format(ep)]
        random_obs_key = list(ep_data_grp["obs"].keys())[0]
        num_samples = len(ep_data_grp["obs/{}".format(random_obs_key)])
        actions = ep_data_grp["actions"][()]
        action_modes = ep_data_grp["action_modes"][()]

        criticals = np.zeros(action_modes.shape)

        grasp_points = []
        release_points = []
        for i in range(len(actions) - 1):
            grip = actions[i][-1]
            grip_next = actions[i + 1][-1]
            if grip < 0 and grip_next > 0:
                grasp_points.append(i + 1)
            if grip > 0 and grip_next < 0:
                release_points.append(i + 1)

        intv_points = []

        pre_intv_len = 35 #args.pre_intv_len * 3
        for i in grasp_points[-1:]:
            for p in range(-12, pre_intv_len - 12):
                if i-p < len(actions):
                    intv_points.append(i-p)

        for i in release_points[-1:]:
            for p in range(-12, pre_intv_len - 12):
                if i-p < len(actions):
                    intv_points.append(i-p)

        print(len(intv_points))
        others = list(range(len(action_modes)))
        nonintv_points = list(set(others) - set(intv_points))
        print(len(nonintv_points))
        print(nonintv_points)

        num_points = args.pre_intv_len * 2
        sampled_indices = random.sample(nonintv_points, num_points)

        for i in sampled_indices:
            criticals[i] = 1

        if "action_modes" in ep_data_grp:
            del ep_data_grp["action_modes"]
        ep_data_grp.create_dataset("action_modes", data=criticals)
        print(ep_data_grp["action_modes"][()])

    print("Done.")
    f.close()
