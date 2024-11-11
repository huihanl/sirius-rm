import os
import h5py
import argparse
import numpy as np

from shutil import copyfile

from tabulate import tabulate

DEMO = -1
ROLLOUT = 0
INTV = 1


names = {
    "-1": "Demo",
    "0": "Rollout",
    "1": "Intv",
    "-10": "Preintv",
    -1: "Demo",
    0: "Rollout",
    1: "Intv",
    -10: "Preintv", 
}

def get_categories_ratio(f, dataset_name):

    demos = sorted(list(f["data"].keys()))

    intv_labels_all = []
    
    for ep in demos:
        # store trajectory
        ep_data_grp = f["data/{}".format(ep)]
        random_obs_key = list(ep_data_grp["obs"].keys())[0]
        num_samples = len(ep_data_grp["obs/{}".format(random_obs_key)])
        intv_labels = ep_data_grp["intv_labels"][()]
        intv_labels_all.extend(intv_labels)
    
    unique, counts = np.unique(intv_labels_all, return_counts=True)
    total = len(intv_labels_all)
    
    unique = [names[u] for u in unique]
    counts_ratio = np.array([c / total for c in counts])
    
    headers = ["Category", "Count", "Percentage"]
    
    data = zip(
        unique + ["Total"],
        list(counts) + [sum(counts)], 
        list(counts_ratio) + [1]
      )
    print("======================================")
    print()
    print("\033[94mDataset: ", dataset_name)
    print()
    print("\033[95m{}".format(tabulate(data, headers=headers)))
    print()
    print("======================================")
    print()
    print()
    
    return unique, counts, counts_ratio


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

    f = h5py.File(out_dataset_path, "r")

    get_categories_ratio(f, args.dataset[:-5])

    demos = sorted(list(f["data"].keys()))

    print()
    print("Adding action modes to:", out_dataset_path, "...")

    total_samples = 0
    total_intv_samples = 0
    total_demo_samples = 0
    total_rollout_samples = 0

    length_lst = []
    intv_length_lst = []
    num_of_intvs = []
    
    succ_rollouts = 0
    for ep in demos:
        # store trajectory
        ep_data_grp = f["data/{}".format(ep)]

        num_samples = len(ep_data_grp["action_modes"][()])
        intv_samples = np.sum(ep_data_grp["action_modes"][()] == 1)
        demo_samples = np.sum(ep_data_grp["action_modes"][()] == -1)

        if intv_samples > 0:
            total_rollout_samples += num_samples

        length_lst.append(num_samples)
        intv_length_lst.append(intv_samples)

        total_samples += num_samples
        total_intv_samples += intv_samples
        total_demo_samples += demo_samples
        if intv_samples == 0:
            succ_rollouts += 1


        action_modes = ep_data_grp["action_modes"][()]
        intv_points = []
        intv_lengths = []
        last_intv = -1
        for i in range(len(action_modes) - 2):
            a = action_modes[i]
            a_next = action_modes[i + 1]
            a_next_next = action_modes[i + 2]
            if a == ROLLOUT and a_next == INTV and a_next_next == INTV: # need at least 2 intv
                intv_points.append(i + 1)
                last_intv = i + 1
            if a == INTV and a_next == ROLLOUT and a_next_next == ROLLOUT:
                intv_length = i - last_intv
                last_intv = -1
                intv_lengths.append(intv_length)

        intv_length_lst.extend(intv_lengths)
        num_of_intvs.append(len(intv_lengths))

    print("===============")
    print("number of trajs: ", len(demos))
    print("total samples: ", total_samples)
    print("total intv samples: ", total_intv_samples)
    print("total demo samples: ", total_demo_samples)

    print("intervention ratio: ", total_intv_samples / total_samples)
    print("intervention ratio in rollouts: ", total_intv_samples / total_rollout_samples)

    print("average traj length: ", sum(length_lst) / len(length_lst))
    print("average intv length: ", sum(intv_length_lst) / len(intv_length_lst))
    print("average number of intv per traj: ", sum(num_of_intvs) / len(num_of_intvs))

    print("succ rollouts: ", succ_rollouts)
    print("succ rollouts ratio: ", succ_rollouts / len(demos))

    print("Done.")
    f.close()
