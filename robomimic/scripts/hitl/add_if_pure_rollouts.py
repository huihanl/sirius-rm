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
        "--name",
        type=str,
        default="pure_rollout",
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

    key_name = "pull_rollout"

    for ep in demos:
        # store trajectory
        ep_data_grp = f["data/{}".format(ep)]
        action_modes = ep_data_grp["action_modes"][()]
        num_samples = len(action_modes)

        property = np.zeros(action_modes.shape)

        all_rollouts = (ep_data_grp["action_modes"][()] == 0).all()
        if all_rollouts:
            property = np.ones(action_modes.shape)

        if key_name in ep_data_grp:
            del ep_data_grp[key_name]
        ep_data_grp.create_dataset(key_name, data=property)
        print(ep_data_grp[key_name][()])

    print("Done.")
    f.close()
