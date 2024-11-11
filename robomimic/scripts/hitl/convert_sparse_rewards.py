import os
import h5py
import argparse
import numpy as np

from shutil import copyfile

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
        "--reward_threshold",
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
    print("Changing dense to sparse reward: ", out_dataset_path, "...")

    success_count = 0
    for ep in demos:
        # store trajectory
        ep_data_grp = f["data/{}".format(ep)]
        rewards = ep_data_grp["rewards"][()]
        new_rewards = (rewards > args.reward_threshold).astype(np.int64)
        #new_rewards = new_rewards_ori.copy()
        #for i in range(len(new_rewards_ori)):
        #    if i < 19 or (new_rewards_ori[i-19:i] == 0).any():
        #        new_rewards[i] = 0

        print(new_rewards)
        success_count += new_rewards[-1]
        del ep_data_grp["rewards"]
        ep_data_grp.create_dataset("rewards", data=new_rewards)

    print("success: {} of total {}".format(success_count, len(demos)))
    print("Done.")
    f.close()
