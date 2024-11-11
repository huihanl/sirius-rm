import os
import h5py
import argparse
import numpy as np

def _get_human_samples_count(f, ep):
    intv_samples = np.sum(f["data"][ep]["action_modes"][()] == 1)
    return intv_samples

def select_human_samples(f, human_sample_num):
    demos = list(f["data"].keys())

    if human_sample_num == -1: # select all
        return demos

    # sort demo keys
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    total_intv = 0
    traj_num = 0
    for i in range(len(demos)):
        traj_num = i
        intv_count = _get_human_samples_count(f, demos[i])
        total_intv += intv_count
        if total_intv > human_sample_num:
            break
    print("# trajs selected: ", traj_num)
    demos = demos[:traj_num]
    return demos

def merge_datasets(datasets, output_dataset, human_samples):
    print()
    print("Merging datasets...")

    src_files = [h5py.File(os.path.expanduser(d), "r") for d in datasets]
    f_out = h5py.File(os.path.expanduser(output_dataset), "w")

    data_grp = f_out.create_group("data")

    total_samples = 0

    demo_num = 0
    for i in range(len(src_files)):
        f_src = src_files[i]
        human_sample_num = human_samples[i]
        demo_key_lst = select_human_samples(f_src, human_sample_num)
        print(demo_key_lst)
        print(len(demo_key_lst))

        for src_demo_id in demo_key_lst:
            demo = f_src["data/{}".format(src_demo_id)]
            target_demo_id = "demo_{}".format(demo_num)
            f_src.copy(demo, data_grp, target_demo_id)

            demo_num += 1

            total_samples += len(demo["actions"]) #demo.attrs['num_samples'] 
            print("src_demo_id {}, total samples {}".format(src_demo_id, total_samples))

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
        "--human_samples",
        type=int,
        nargs='+',
        default=[-1, -1]
    )

    args = parser.parse_args()

    assert args.output_dataset not in args.datasets

    merge_datasets(args.datasets, args.output_dataset, args.human_samples)
