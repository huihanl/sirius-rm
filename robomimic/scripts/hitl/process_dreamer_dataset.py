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
        "--pre_intv_len",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--neg",
        action="store_true",
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

    def find_different_indices(tensor1, tensor2):
        # Check if the tensors have the same shape
        if tensor1.shape != tensor2.shape:
            assert Error

        # Create a boolean mask of elements that are different
        mask = tensor1 != tensor2

        # Find the indices of the different elements
        indices = np.nonzero(mask)
        print(indices)

        # If there are no different elements, return None
        if indices[0].shape[0] == 0:
            return None

        # Return the indices where the tensors differ
        return indices

    for ep in demos:
        # store trajectory
        ep_data_grp = f["data/{}".format(ep)]
        random_obs_key = list(ep_data_grp["obs"].keys())[0]
        num_samples = len(ep_data_grp["obs/{}".format(random_obs_key)])
        actions = ep_data_grp["actions"][()]

        actions_copy = actions.copy()

        action_dim = actions_copy.shape[1]
        zero_action = np.zeros((1, action_dim))
        if args.neg:
            zero_action[0,-1] = -1.
        prev_actions = np.concatenate((zero_action, actions_copy[:-1]), axis=0)

        inds = find_different_indices(prev_actions[1:], actions[:-1])
        assert inds is None

        ############

        if "prev_actions" in ep_data_grp:
            del ep_data_grp["prev_actions"]
        ep_data_grp.create_dataset("prev_actions", data=prev_actions)
        # print(ep_data_grp["prev_actions"][()])

        ############

        is_first = np.zeros(ep_data_grp["action_modes"].shape)
        is_first[0] = 1.

        if "is_first" in ep_data_grp:
            del ep_data_grp["is_first"]
        ep_data_grp.create_dataset("is_first", data=is_first)
        print(ep_data_grp["is_first"][()])

    print("Done.")
    f.close()
