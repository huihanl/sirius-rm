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
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
    )

    args = parser.parse_args()

    out_dataset_path = os.path.expanduser(args.dataset)

    f = h5py.File(out_dataset_path, "r+")

    demos = sorted(list(f["data"].keys()))

    print()
    print("Adding action modes to:", out_dataset_path, "...")

    actions_all = []

    for ep in demos:
        # store trajectory
        ep_data_grp = f["data/{}".format(ep)]
        actions = ep_data_grp["actions"][()]
        actions_all.extend(list(actions))

    actions_all = np.array(actions_all)
    print((actions_all >= -1.0).all())
    print((actions_all <= 1.0).all())
    print((actions_all > -1.0).all())
    print((actions_all < 1.0).all())

    print("Done.")
    f.close()
