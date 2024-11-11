import h5py
import numpy as np
from shutil import copyfile

dataset = '/home/soroushn/research/robomimic-dev/datasets/hitl/faucet_real/0120_faucet.hdf5'
new_dataset = dataset[:-5] + '_trunc.hdf5'

offset = 100

copyfile(dataset, new_dataset)
f = h5py.File(new_dataset, "r+")

def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

total_num_samples = 0

for demo_id in list(f['data'].keys()):
    ep_grp = f['data/{}'.format(demo_id)]

    actions = ep_grp['actions'][()]
    inds = np.where(actions == 1)
    first_ind = inds[0][0]
    trunc_len = first_ind + offset

    dataset_keys = get_dataset_keys(ep_grp)
    for k in dataset_keys:
        vals = ep_grp[k][()]
        del ep_grp[k]
        ep_grp.create_dataset(k, data=vals[:trunc_len])

    ep_grp.attrs['num_samples'] = trunc_len
    total_num_samples += trunc_len

f['data'].attrs['total'] = total_num_samples

f.close()