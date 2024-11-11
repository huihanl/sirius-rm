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

    total = 0
    for ep in demos:
        # store trajectory
        ep_data_grp_obs = f["data/{}/obs".format(ep)]
        
        joint_states = ep_data_grp_obs["joint_states"][()]
        joint_states = np.round(joint_states, 3)
        length = 5
        alr = False
        delete_pos = []
        for i in range(joint_states.shape[0]-length):
            joint_seq = joint_states[i:i+length]
            if (joint_seq == joint_seq[0]).all():
                print(ep, i)
                delete_pos.extend(list(range(i, i + length - 1)))
        delete_pos = list(set(delete_pos))
        
        if len(delete_pos) > 0:
            for key in ep_data_grp_obs:
                entry = ep_data_grp_obs[key][()]
                entry = np.delete(entry, delete_pos, 0)
                del ep_data_grp_obs[key]
                ep_data_grp_obs[key] = entry
            data_ = f["data/{}".format(ep)]
            for key in data_:
                if key == "obs":
                    continue
                entry = data_[key][()]
                entry = np.delete(entry, delete_pos, 0)
                del data_[key]
                data_[key] = entry
            assert data_["actions"][()].shape[0] == \
                    data_["action_modes"][()].shape[0] == \
                    data_["obs"]["joint_states"][()].shape[0]
            f["data/{}".format(ep)].attrs["num_samples"] = data_["actions"][()].shape[0]
        total += f["data/{}".format(ep)].attrs["num_samples"]
        
    f["data"].attrs["total"] = total
    print("total samples: ", total)
    print("Done.")
    f.close()
