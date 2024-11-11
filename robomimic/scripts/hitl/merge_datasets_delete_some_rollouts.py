import os
import h5py
import argparse
import numpy as np
import random

def get_deleted_ep_idx(f, delete_num):
    if delete_num == -1: # do not delete
        return []

    # Sort demo keys
    demos = list(f["data"].keys())
    random.shuffle(demos)

    # Select ${delete_num} number of trajectories to delete
    delete_lst = []
    for ep in demos:
        ep_data_grp = f["data/{}".format(ep)]
        if (ep_data_grp["action_modes"][()] == 0).all():
            delete_lst.append(ep)
        if len(delete_lst) == delete_num:
            break

    return delete_lst

def merge_datasets(dataset, output_dataset, delete_num):
    print()
    print("Merging datasets...")

    f_in = h5py.File(os.path.expanduser(dataset), "r")
    f_out = h5py.File(os.path.expanduser(output_dataset), "w")

    data_grp = f_out.create_group("data")

    total_samples = 0

    delete_eps = get_deleted_ep_idx(f=f_in, delete_num=delete_num)
    print(len(delete_eps))

    for ep in delete_eps:
        demo = f_in["data/{}".format(ep)]
        assert (demo["action_modes"][()] == 0).all()

    all_eps = list(f_in["data"].keys())
    saved_eps = list(set(all_eps) - set(delete_eps))

    #############################################################
    intv_original = 0
    for ep in all_eps:
        demo = f_in["data/{}".format(ep)]
        intv_samples = np.sum(demo["action_modes"][()] == 1)
        intv_original += intv_samples

    intv_now = 0
    for ep in saved_eps:
        demo = f_in["data/{}".format(ep)]
        intv_samples = np.sum(demo["action_modes"][()] == 1)
        intv_now += intv_samples

    print("original intv: ", intv_original)
    print("now intv: ", intv_now)
    #############################################################

    for demo_num in range(len(saved_eps)):
        src_demo_id = saved_eps[demo_num]
        demo = f_in["data/{}".format(src_demo_id)]
        target_demo_id = "demo_{}".format(demo_num)
        f_in.copy(demo, data_grp, target_demo_id)

        total_samples += len(demo["actions"])
        #print("src_demo_id {}, total samples {}".format(src_demo_id, total_samples))

    intv_new = 0
    for ep in data_grp:
        demo = data_grp["{}".format(ep)]
        intv_samples = np.sum(demo["action_modes"][()] == 1)
        intv_new += intv_samples

    print("original intv: ", intv_original)
    print("now intv: ", intv_now)
    print("intv for new dataset: ", intv_new)

    data_grp.attrs["total"] = total_samples
    if "env_args" in f_in["data"].attrs:
        data_grp.attrs["env_args"] = f_in["data"].attrs["env_args"]

    data_grp.attrs["dataset"] = [os.path.expanduser(dataset)]

    f_in.close()
    f_out.close()

    print("Done.")

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

    parser.add_argument(
        "--delete-num",
        type=int,
        default=-1,
    )

    args = parser.parse_args()

    assert args.output_dataset != args.dataset

    merge_datasets(args.dataset, args.output_dataset, args.delete_num)
