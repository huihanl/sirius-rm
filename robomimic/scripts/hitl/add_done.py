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
        "--done_mode",
        type=int,
        default=2, # done = reward by default
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
        rewards = ep_data_grp["rewards"][()]
        if args.done_mode == 2:
            dones = np.array(rewards).copy()
        elif args.done_mode == 1:
            num_samples = len(rewards)
            dones = np.zeros(shape=(num_samples,), dtype=np.int64)
            dones[-1] = rewards[-1] 
        print(dones)
        del ep_data_grp["dones"]
        ep_data_grp.create_dataset("dones", data=dones)

    print("Done.")
    f.close()
