import os
import h5py
import argparse
import numpy as np

from shutil import copyfile

"""
DEMO = -1
ROLLOUT = 0
INTV = 1
FULL_ROLLOUT = 2
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
        ep_data_grp_obs = f["data/{}/obs".format(ep)]
        
        gripper_states = ep_data_grp_obs["gripper_states"][()]
        if len(gripper_states.shape) == 3 and gripper_states.shape[1] == 1 and gripper_states.shape[2] == 1:
            print(ep_data_grp_obs["gripper_states"].shape)
            gripper_states = np.squeeze(gripper_states, axis=2)
            del ep_data_grp_obs["gripper_states"]
            ep_data_grp_obs.create_dataset("gripper_states", data=gripper_states)
            print(ep_data_grp_obs["gripper_states"].shape)

    print("Done.")
    f.close()
