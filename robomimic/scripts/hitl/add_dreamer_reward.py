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
        intv_label = ep_data_grp["intv_labels"][()]
        
        sparse_reward = np.zeros_like(intv_label)
        sparse_reward[intv_label == -10] = -1
        three_class = sparse_reward.copy()
        three_class[intv_label == 1] = -2

        dense_reward = np.zeros_like(intv_label)
        intv_points = {}
        for i in range(len(intv_label) - 2):
            a = action_modes[i]
            a_next = action_modes[i + 1]
            a_next_next = action_modes[i + 2]
            if a == ROLLOUT and a_next == INTV and a_next_next == INTV: # need at least 2 intv
                intv_point = i + 1

                # count number of intv for reward
                intv_count = 0
                for j in range(i + 1, len(intv_label)):
                    if action_modes[j] == INTV:
                        intv_count += 1
                    else:
                        break
                intv_points[intv_point] = intv_count

        def _compute_reward(idx):
            reward = -1 * idx
            return reward

        for intv_point in intv_points:
            intv_count = intv_points[intv_point]
            for p in range(1, intv_count):
                if action_modes[intv_point-p] != INTV:
                    dense_reward[intv_point-p] = _compute_reward(intv_count - p)
                else:
                    break

        assert (intv_label[dense_reward < 0] != INTV).all()
        
        if "dense_reward" in ep_data_grp:
            del ep_data_grp["dense_reward"] 
        ep_data_grp.create_dataset("dense_reward", data=dense_reward)
        
        if "three_class" in ep_data_grp:
            del ep_data_grp["three_class"]
        ep_data_grp.create_dataset("three_class", data=three_class)

        #print(ep_data_grp["dense_reward"][()])

        if "sparse_reward" in ep_data_grp:
            del ep_data_grp["sparse_reward"] 
        ep_data_grp.create_dataset("sparse_reward", data=sparse_reward)
        print(ep_data_grp["sparse_reward"][()])

        all_sample_count += len(ep_data_grp["sparse_reward"][()])
        failure_count -= sum(ep_data_grp["sparse_reward"][()])
    
    print("Failure ratio: ", failure_count / all_sample_count)
    print("failure: ", failure_count)
    print("all: ", all_sample_count)

    print("Done.")
    f.close()
