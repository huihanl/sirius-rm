import os
import h5py
import argparse
import numpy as np

def select_traj_num(f, traj_num, shuffle):
    demos = list(f["data"].keys())

    if traj_num == -1: # select all
        return demos

    # sort demo keys
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if shuffle:
        import random
        random.shuffle(demos)
    
    demos = demos[:traj_num]
    return demos

def merge_datasets(datasets, output_dataset, num_lst, shuffle):
    print()
    print("Merging datasets...")

    src_files = [h5py.File(os.path.expanduser(d), "r") for d in datasets]
    f_out = h5py.File(os.path.expanduser(output_dataset), "w")

    data_grp = f_out.create_group("data")

    total_samples = 0

    demo_num = 0
    
    assert len(src_files) == 1
    
    for i in range(len(src_files)):
        f_src = src_files[i]
        traj_num = num_lst[i]
        
        demo_key_lst = src_files[i]["mask/round3"]
        
        print(demo_key_lst)
        print(len(demo_key_lst))

        for src_demo_id in demo_key_lst:
            
            # only get data with intervention
            action_modes = f_src["data/{}".format(src_demo_id)]["action_modes"][()]
            if (action_modes == -1).all() or (action_modes == 0).all():
                continue
            
            demo = f_src["data/{}".format(src_demo_id)]
            target_demo_id = "demo_{}".format(demo_num)
            f_src.copy(demo, data_grp, target_demo_id)

            demo_num += 1

            total_samples += len(demo["actions"]) #demo.attrs['num_samples'] 
            print("src_demo_id {}, total samples {}".format(src_demo_id, total_samples))

    print("total number of trajectories: ", demo_num)

    data_grp.attrs["total"] = total_samples
    if "env_args" in src_files[0]["data"].attrs:
        data_grp.attrs["env_args"] = src_files[0]["data"].attrs["env_args"]

    data_grp.attrs["datasets"] = [os.path.expanduser(d) for d in datasets]

    for f_src in src_files:
        f_src.close()
    f_out.close()

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
    )

    parser.add_argument(
        "--num_lst",
        type=int,
        nargs='+',
        default=[-1, -1]
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
    )

    args = parser.parse_args()

    assert args.output_dataset not in args.datasets

    merge_datasets(args.datasets, args.output_dataset, args.num_lst, args.shuffle)
