import os
import h5py
import argparse
import numpy as np

from shutil import copyfile

"""
DEMO = 1
#PRE_ROLLOUT = -1
ROLLOUT = 0
INTV = 2
"""

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

    for ep in demos:
        # store trajectory
        ep_data_grp = f["data/{}".format(ep)]
        random_obs_key = list(ep_data_grp["obs"].keys())[0]
        num_samples = len(ep_data_grp["obs/{}".format(random_obs_key)])
        action_modes = ep_data_grp["action_modes"]

        rewards = [0] * len(list(ep_data_grp["rewards"][()]))
        terminals = ep_data_grp["dones"][()]
        
        mistake_idx = []
        for i in range(len(action_modes) - 1):
            if action_modes[i] == 0 and action_modes[i + 1] == 1:
                mistake_idx.append(i)

        for idx in mistake_idx:
            rewards[idx] = -1
            terminals[idx] = 1

        """
        for idx in mistake_idx:
            i = 1
            while idx + i < len(action_modes) and action_modes[idx + i] == 1:
                i += 1
            #rewards[idx + i - 1] = 30
            rewards[i] = 10
            for c in range(i-1, idx, -1):
                rewards[c] = rewards[c+1] - 10 / (i - idx)
        """

        print("rewards: ")
        print(rewards)

        print("terminals: ")
        print(terminals)

        del ep_data_grp["rewards"]
        ep_data_grp.create_dataset("rewards", data=rewards)
        
        del ep_data_grp["dones"]
        ep_data_grp.create_dataset("dones", data=terminals)
        

    print("Done.")
    f.close()
