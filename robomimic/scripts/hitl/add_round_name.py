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
        default="round",
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

    key_name = args.name

    mask_names = f["mask"].keys()
    name_dict = {}
    for name in mask_names:
        name_group = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(name)])]
        name_dict[name] = set(name_group)

    for ep in demos:
        # store trajectory
        ep_data_grp = f["data/{}".format(ep)]
        action_modes = ep_data_grp["action_modes"][()]
        num_samples = len(action_modes)

        round_names = None
        
        for name in name_dict:
            if ep in name_dict[name]:
                name_int = int(name[-1])
                round_names = np.array([name_int] * num_samples)
        assert round_names is not None

        if key_name in ep_data_grp:
            del ep_data_grp[key_name]
        ep_data_grp.create_dataset(key_name, data=round_names)
        print(ep_data_grp[key_name][()])

    print("Done.")
    f.close()
