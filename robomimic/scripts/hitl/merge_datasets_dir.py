import os
import h5py
import argparse

def merge_datasets(datasets, output_dataset):
    print()
    print("Merging datasets...")

    src_files = [h5py.File(os.path.expanduser(d), "r") for d in datasets]
    f_out = h5py.File(os.path.expanduser(output_dataset), "w")

    data_grp = f_out.create_group("data")

    total_samples = 0

    demo_num = 0
    for f_src in src_files:
        for src_demo_id in list(f_src["data"].keys()):
            demo = f_src["data/{}".format(src_demo_id)]
            target_demo_id = "demo_{}".format(demo_num)
            f_src.copy(demo, data_grp, target_demo_id)

            demo_num += 1

        total_samples += f_src["data"].attrs["total"]

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
        "--datasets_dir",
        type=str,
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
    )
    args = parser.parse_args()

    #assert args.output_dataset not in args.datasets
    
    datasets = []

    for root, dirs, files in os.walk(args.datasets_dir):
        for f in files:
            if ".hdf5" in f:
                f_path = os.path.join(root, f)
                print(f_path)
                datasets.append(f_path)

    merge_datasets(datasets, args.output_dataset)
