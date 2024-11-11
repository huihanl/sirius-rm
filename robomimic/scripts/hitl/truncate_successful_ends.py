import h5py
import numpy as np
from shutil import copyfile
import sys

dataset = sys.argv[1]
print(dataset)

offset = int(sys.argv[2])

new_dataset = dataset[:-5] + '_trunc_{}.hdf5'.format(offset)

copyfile(dataset, new_dataset)
f = h5py.File(new_dataset, "r+")

def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

total_num_samples = 0

for demo_id in list(f['data'].keys()):
    ep_grp = f['data/{}'.format(demo_id)]

    rewards = ep_grp['rewards'][()]
    inds = np.where(rewards == 1)
    if inds[0].size == 0:
        continue
    first_ind = inds[0][0]
    remain_len = len(rewards) - first_ind
    true_len = min(offset, remain_len)
    trunc_len = first_ind + true_len
    print(trunc_len)

    dataset_keys = get_dataset_keys(ep_grp)
    for k in dataset_keys:
        vals = ep_grp[k][()]
        del ep_grp[k]
        ep_grp.create_dataset(k, data=vals[:trunc_len])

    ep_grp.attrs['num_samples'] = trunc_len
    total_num_samples += trunc_len

f['data'].attrs['total'] = total_num_samples

f.close()
