"""
A script to visualize dataset trajectories by loading the simulation states
one by one or loading the first state and playing actions back open-loop.
The script can generate videos as well, by rendering simulation frames
during playback. The videos can also be generated using the image observations
in the dataset (this is useful for real-robot datasets) by using the
--use-obs argument.

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, use the subset of trajectories
        in the file that correspond to this filter key

    n (int): if provided, stop after n trajectories are processed

    use-obs (bool): if flag is provided, visualize trajectories with dataset 
        image observations instead of simulator

    use-actions (bool): if flag is provided, use open-loop action playback 
        instead of loading sim states

    render (bool): if flag is provided, use on-screen rendering during playback
    
    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    render_image_names (str or [str]): camera name(s) / image observation(s) to 
        use for rendering on-screen or to video

    first (bool): if flag is provided, use first frame of each episode for playback
        instead of the entire episode. Useful for visualizing task initializations.

Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_image_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-actions --render_image_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

import os
import json
import h5py
import argparse
import imageio
import numpy as np
import math

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType

import robomimic.utils.vis_utils as VisUtils
import robomimic.utils.tensor_utils as TensorUtils
            
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}


def prepare_tensor(tensor, device=None):
    """
    Prepare raw observation dict from environment for policy.

    Args:
        ob (dict): single observation dictionary from environment (no batch dimension, 
            and np.array values for each key)
    """
    tensor = TensorUtils.to_tensor(tensor)
    tensor = TensorUtils.to_batch(tensor)
    if device is not None:
        tensor = TensorUtils.to_device(tensor, device)
    tensor = TensorUtils.to_float(tensor)
    return tensor

def _check_grasp_success(env):
    grasped = env.env._check_grasp(
                  gripper=env.env.robots[0].gripper,
                  object_geoms=[g for g in env.env.nuts[0].contact_geoms])
    return grasped

def playback_trajectory_with_env(
    env, 
    initial_state, 
    states, 
    actions=None, 
    algo=None,
    render=False, 
    video_writer=None, 
    video_skip=5, 
    camera_names=None,
    first=False,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert isinstance(env, EnvBase)

    write_video = (video_writer is not None)
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    traj_len = states.shape[0]

    adv_vals = []
    v_vals = []
    q_vals = []
    grasp_success_t = []

    for i in range(traj_len):
        env.reset_to({"states" : states[i]})

        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])
            
        ob = env.get_observation()
        ob = prepare_tensor(ob, device=algo.device)
        ac = prepare_tensor(actions[i], device=algo.device)
        
        v_value = algo.get_v_value(obs_dict=ob)
        v_value = np.around(v_value.item(), decimals=3)
        v_vals.append(v_value)
        
        adv_value = algo.get_adv_weight(obs_dict=ob, ac=ac)
        adv_value = np.around(adv_value.item(), decimals=3)
        adv_vals.append(adv_value)

        q_value = algo.get_Q_value(obs_dict=ob, ac=ac)
        q_value = np.around(q_value.item(), decimals=3)
        q_vals.append(q_value)

        if len(grasp_success_t) == 0 and _check_grasp_success(env):
            grasp_success_t.append(i)

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    proc_img = env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name)
                    text = "A: {}\nV: {:.2e}".format(adv_value, v_value)
                    proc_img = VisUtils.write_text_on_image(proc_img, text)
                    video_img.append(proc_img)
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                video_writer.append_data(video_img)
            video_count += 1

        if first:
            break
            
    return dict(v_vals=v_vals, 
                adv_vals=adv_vals, 
                q_vals=q_vals,
                grasp_success_t=grasp_success_t,
                )


def playback_dataset(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[args.n : args.n + 10]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)
    
    algo, ckpt_dict = FileUtils.algo_from_checkpoint(ckpt_path=args.model)
    
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )
    
    is_robosuite_env = True
    
    if len(demos) >= 5:
        num_cols = 5
        num_rows = int(math.ceil(len(demos) / 5))
    else:
        num_cols = len(demos)
        num_rows = 1
        
    # plt.figure(figsize=(num_cols * 3, num_rows * 3))

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        actions = f["data/{}/actions".format(ep)][()]
        action_modes = f["data/{}/action_modes".format(ep)][()]

        vals = playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state, 
            states=states, actions=actions, 
            algo=algo,
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
        )
   
        rewards = f["data/{}/rewards".format(ep)][()]
        success = rewards[-1]

        #v_vals=v_vals, adv_vals=adv_vals, q_vals=q_vals
        plot_twin(ind=ind, 
                  data0=vals["adv_vals"], 
                  name0="Adv Weights", 
                  data1=vals["v_vals"], 
                  name1="V Value", 
                  vert_info=vals["grasp_success_t"],
                  background_key=action_modes,
                  success=success,
                  )

        plot_twin(ind=ind,
                  data0=vals["q_vals"],
                  name0="Q value",
                  data1=vals["v_vals"],
                  name1="V Value",
                  vert_info=vals["grasp_success_t"],
                  background_key=action_modes,
                  success=success,
                  )

        v_delta = np.array(vals["v_vals"][1:]) - np.array(vals["v_vals"][:-1])
        plot_twin(ind=ind, 
                  data0=vals["adv_vals"], 
                  name0="Adv Weights", 
                  data1=v_delta, 
                  name1="V Delta", 
                  lim=5,
                  background_key=action_modes,
                  )
            
    f.close()
    if write_video:
        video_writer.close()

def plot_twin(ind, data0, name0, data1, name1, vert_info=[], lim=None, background_key=None, success=None):
    """
    Plotting code from https://www.geeksforgeeks.org/use-different-y-axes-on-the-left-and-right-of-a-matplotlib-plot/
    """
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel(name0, color = color)
    ax1.plot(data0, color = color)
    ax1.tick_params(axis ='y', labelcolor = color)
    ax1.axhline(y = 0.0, color = 'black')
    if success is not None:
        ax1.title.set_text('Success' if success else "Failure")

    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel(name1, color = color)
    ax2.plot(data1, color = color)
    ax2.tick_params(axis ='y', labelcolor = color)
    ax2.axhline(y = 0.0, color = 'black')

    for i in vert_info:
        ax2.axvline(x = i, color = 'yellow')

    if background_key is not None:
        color_inds = np.reshape(np.argwhere(background_key == 1), -1) 
        for i in color_inds:
            ax2.axvline(x=i, color='green', linewidth=5, alpha=0.03)

    if lim is not None:
        ax1.set_ylim(-lim, lim)
        ax2.set_ylim(-lim, lim)

    plt.savefig(os.path.join(
        os.path.dirname(args.video_path),
        'plot_{}_{}_{}.png'.format(ind, name0.replace(" ", "_"), name1.replace(" ", "_"))
    ))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="path to trained model (AWAC or IQL supported)",
    )
    
    # /home/soroushn/research/hitl/expdata/hitl/square/ld/iql/debug/quant_0.9_beta_3.0_igndone_True_discount_0.99/2022-02-19-17-26-34/models/model_epoch_2_NutAssemblySquare_success_0.0.pth
    # /home/soroushn/research/hitl/expdata/hitl/square/ld/awac/debug/ds_hitl-soroush_beta_3.0_igndone_True_discount_0.99/2022-02-19-18-14-51/models/model_epoch_2_NutAssemblySquare_success_0.0.pth
    
    
    
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # # Use image observations instead of doing playback using the simulator env.
    # parser.add_argument(
    #     "--use-obs",
    #     action='store_true',
    #     help="visualize trajectories with dataset image observations instead of simulator",
    # )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    args = parser.parse_args()

    # check if video path exist
    video_dir = os.path.dirname(args.video_path)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    playback_dataset(args)
