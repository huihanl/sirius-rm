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
    parser.add_argument(
        "--value",
        type=int,
        required=True,
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
        action_modes = args.value * np.ones(shape=(num_samples,), dtype=np.int64)
        print(action_modes)
        if "action_modes" not in ep_data_grp:
            ep_data_grp.create_dataset("action_modes", data=action_modes)

    print("Done.")
    f.close()
