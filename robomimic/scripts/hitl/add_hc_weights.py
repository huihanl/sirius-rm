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
        "--pre_intv_len",
        type=int,
        default=15,
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
        action_modes = ep_data_grp["action_modes"][()]
        intv_label = action_modes.copy()
        intv_points = []
        for i in range(len(action_modes) - 2):
            a = action_modes[i]
            a_next = action_modes[i + 1]
            a_next_next = action_modes[i + 2]
            if a == ROLLOUT and a_next == INTV and a_next_next == INTV: # need at least 2 intv
                intv_points.append(i + 1)
  
        pre_intv_len = args.pre_intv_len
        for i in intv_points:
            for p in range(pre_intv_len):
                if action_modes[i-p] != INTV:
                    intv_label[i-p] = PRE_INTV

        if "intv_labels" in ep_data_grp:
            del ep_data_grp["intv_labels"] 
        ep_data_grp.create_dataset("intv_labels", data=intv_label)
        print(ep_data_grp["intv_labels"][()])

    print("Done.")
    f.close()
