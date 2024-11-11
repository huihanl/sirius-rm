import h5py
import numpy as np
from shutil import copyfile
import sys
import os

dataset = sys.argv[1]
print(dataset)
dataset_succ = dataset[:-5] + '_rollout.hdf5'
dataset_fail = dataset[:-5] + '_others.hdf5'

f_original = h5py.File(os.path.expanduser(dataset), "r")

f_succ = h5py.File(os.path.expanduser(dataset_succ), "w")
f_fail = h5py.File(os.path.expanduser(dataset_fail), "w")

def split_datasets():
    print()
    print("Splitting success and failure trajectories ...")

    data_grp_succ = f_succ.create_group("data")
    data_grp_fail = f_fail.create_group("data")

    total_samples_succ = 0
    total_samples_fail = 0

    demo_num_succ = 0
    demo_num_fail = 0

    for src_demo_id in list(f_original["data"].keys()):
        demo = f_original["data/{}".format(src_demo_id)]
        rewards = demo['rewards'][()]
        num_samples = rewards.shape[0]
        
        inds = np.where(rewards == 1)

        action_modes = demo['action_modes'][()]

        is_success = (action_modes == 0).all() 

        if is_success:
            target_demo_id = "demo_{}".format(demo_num_succ)
            f_original.copy(demo, data_grp_succ, target_demo_id)
            demo_num_succ += 1
            total_samples_succ += num_samples
        else:
            target_demo_id = "demo_{}".format(demo_num_fail)
            f_original.copy(demo, data_grp_fail, target_demo_id)
            demo_num_fail += 1
            total_samples_fail += num_samples
        print("success count: {}, failure count: {}".format(demo_num_succ, demo_num_fail))

    f_succ['data'].attrs['total'] = total_samples_succ
    f_fail['data'].attrs['total'] = total_samples_fail

    f_succ['data'].attrs['num_traj'] = demo_num_succ
    f_fail['data'].attrs['num_traj'] = demo_num_fail

    print("successful num of demos: ", f_succ['data'].attrs['num_traj'])
    print("failed num of demos: ", f_fail['data'].attrs['num_traj'])

    if "env_args" in f_original["data"].attrs:
        data_grp_succ.attrs["env_args"] = f_original["data"].attrs["env_args"]
    data_grp_succ.attrs["dataset"] = os.path.expanduser(dataset)

    if "env_args" in f_original["data"].attrs:
        data_grp_fail.attrs["env_args"] = f_original["data"].attrs["env_args"]
    data_grp_fail.attrs["dataset"] = os.path.expanduser(dataset)

    f_original.close()
    f_succ.close()
    f_fail.close()
    print("Done.")

split_datasets()
