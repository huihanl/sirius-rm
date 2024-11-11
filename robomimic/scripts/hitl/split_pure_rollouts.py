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

from robomimic.utils.file_utils import create_hdf5_filter_key

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

    total_samples = 0
    total_intv_samples = 0
    total_demo_samples = 0
    succ_rollouts = []
    for ep in demos:
        # store trajectory
        ep_data_grp = f["data/{}".format(ep)]
        random_obs_key = list(ep_data_grp["obs"].keys())[0]

        num_samples = len(ep_data_grp["obs/{}".format(random_obs_key)])
        intv_samples = np.sum(ep_data_grp["action_modes"][()] == 1)
        demo_samples = np.sum(ep_data_grp["action_modes"][()] == -1)

        if intv_samples == 0 and demo_samples == 0:
            succ_rollouts.append(ep)

        total_samples += num_samples
        total_intv_samples += intv_samples
        total_demo_samples += demo_samples

    #print("===============")
    #print("total samples: ", total_samples)
    #print("total intv samples: ", total_intv_samples)
    #print("total demo samples: ", total_demo_samples)
    #print("intervention ratio: ", total_intv_samples / total_samples)
    #print("succ rollout ratio: ", succ_rollouts / len(demos))

    intv_group = list(set(demos) - set(succ_rollouts))
    print(succ_rollouts)
    print(intv_group)
    print("succ rount: ", len(succ_rollouts))
    print("intv count: ", len(intv_group))

    create_hdf5_filter_key(hdf5_path=out_dataset_path, demo_keys=succ_rollouts, key_name="succ_rollouts")
    create_hdf5_filter_key(hdf5_path=out_dataset_path, demo_keys=intv_group, key_name="demos_and_intv")

    print("Done.")
    f.close()
