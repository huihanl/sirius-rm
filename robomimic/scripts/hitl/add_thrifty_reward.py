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

    all_sample_count = 0
    failure_count = 0

    for ep in demos:
        # store trajectory
        ep_data_grp = f["data/{}".format(ep)]
        random_obs_key = list(ep_data_grp["obs"].keys())[0]
        num_samples = len(ep_data_grp["obs/{}".format(random_obs_key)])
        action_modes = ep_data_grp["action_modes"][()]
        rewards = ep_data_grp["rewards"][()]
        if (rewards[-5:] == 1).all():
            if (action_modes == -1).all():
                rewards = np.zeros(rewards.shape)
                rewards[-1] = 1.
                print(rewards)

        #print(ep_data_grp["rewards"][()])
        if len(rewards) < 380:
            rewards[-1] = 1

        del ep_data_grp["rewards"]
        ep_data_grp.create_dataset("rewards", data=rewards)
        failure_count += rewards[-1]
    
    print("Success ratio: ", failure_count / len(demos))
    print("success: ", failure_count)
    print("all: ", all_sample_count)

    print("Done.")
    f.close()
