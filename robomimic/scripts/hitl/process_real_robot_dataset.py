import os
import h5py
import argparse
import numpy as np

from shutil import copyfile

"""
Processes hdf5 data collected on real robot
Adds obs / next_obs / action(processed) entries
Does not add reward / dones, those need to be added later
"""

def postprocess_raw_actions(actions,
                        use_yaw=False,
                        use_ori=False,
                        use_gripper=False):
    if use_ori:
        return actions

    new_actions = [None for i in range(len(actions))]
    for i in range(len(new_actions)):
        new_actions[i] = actions[i][:3]
        assert len(new_actions[i]) == 3
        if use_yaw:
            new_actions[i] = np.append(new_actions[i], [actions[i][-2]])
        if use_gripper:
            new_actions[i] = np.append(new_actions[i], [actions[i][-1]])
        assert len(new_actions[i]) == 3 + int(use_yaw) + int(use_gripper)
    return new_actions

def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]

def add_to_obs_groups(obs_grp, next_obs_grp, key, data):
    obs_grp.create_dataset(key, data=np.array(data[:-1]))
    next_obs_grp.create_dataset(key, data=np.array(data[1:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )

    parser.add_argument(
        "--output_dataset",
        type=str,
        help="path to hdf5 dataset (optional)",
        default=None,
    )

    parser.add_argument(
        "--use_ori",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--use_yaw",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--use_gripper",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--use_gripper_history",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    if args.use_gripper_history:
        # make sure to always create a new dataset in this case
        assert args.output_dataset is not None

    if args.output_dataset is not None:
        assert args.output_dataset != args.dataset
        copyfile(args.dataset, args.output_dataset)
        out_dataset_path = os.path.expanduser(args.output_dataset)
    else:
        out_dataset_path = os.path.expanduser(args.dataset)

    print()
    print("Processing real robot dataset:", out_dataset_path, "...")

    f = h5py.File(out_dataset_path, "r+")

    total_samples = 0

    for demo_key in list(f['data'].keys()):
        ep_data_grp = f['data/{}'.format(demo_key)]

        # read raw actions
        if 'actions_raw' in ep_data_grp:
            actions_raw = np.array(ep_data_grp['actions_raw'])
        else:
            actions_raw = np.array(ep_data_grp['actions'])
            ep_data_grp.move('actions', 'actions_raw')

        # delete existing data (fresh start)
        for k in ["actions", "obs", "next_obs"]:
            if k in ep_data_grp:
                del ep_data_grp[k]

        actions = postprocess_raw_actions(
            actions_raw,
            use_ori=args.use_ori,
            use_yaw=args.use_yaw,
            use_gripper=args.use_gripper,
        )
        actions = actions[:-1] # truncate last step
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        num_samples = len(actions)
        ep_data_grp.attrs["num_samples"] = num_samples
        total_samples += num_samples

        # add obs / next_obs groups
        obs_grp = ep_data_grp.create_group("obs")
        next_obs_grp = ep_data_grp.create_group("next_obs")

        # 3rd person camera
        images_3p = np.array(ep_data_grp['camera_0_color'])
        images_3p = np.transpose(images_3p, (0, 2, 3, 1))
        add_to_obs_groups(obs_grp, next_obs_grp, "agentview_image", images_3p)

        # eye in hand camera
        images_eih = np.array(ep_data_grp['camera_1_color'])
        images_eih = np.transpose(images_eih, (0, 2, 3, 1))
        add_to_obs_groups(obs_grp, next_obs_grp, "robot0_eye_in_hand_image", images_eih)

        # ee pos
        ee_raw = np.array(ep_data_grp['proprio_ee'])
        ee_pos = ee_raw[:,12:15]
        add_to_obs_groups(obs_grp, next_obs_grp, "robot0_eef_pos", ee_pos)

        ## ee quat
        ee_rotmat = ee_raw.reshape((-1, 4, 4))[:, :3, :3]
        ee_quat = np.array([mat2quat(elem) for elem in ee_rotmat])
        add_to_obs_groups(obs_grp, next_obs_grp, "robot0_eef_quat", ee_quat)

        ## gripper
        ee_gripper = np.array(ep_data_grp['proprio_gripper_state'])
        # if using history of gripper
        if args.use_gripper_history:
            gripper_states_list = []
            for j in range(ee_gripper.shape[0]):
                gripper_state = []
                for k in range(j - 5, j):
                    if k < 0:
                        gripper_state += ee_gripper[0].tolist()
                    else:
                        gripper_state += ee_gripper[k].tolist()
                gripper_states_list.append(gripper_state)
            gripper_states_list = np.array(gripper_states_list)
            add_to_obs_groups(obs_grp, next_obs_grp, "robot0_gripper_qpos", gripper_states_list)
        else:
            add_to_obs_groups(obs_grp, next_obs_grp, "robot0_gripper_qpos", ee_gripper)

    f['data'].attrs['total'] = total_samples

    print("Done.")

    f.close()

