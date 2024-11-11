import os
import json
import h5py
import argparse
import imageio
import numpy as np
import time
import cv2
from os.path import exists

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType
from robomimic.scripts.vis.image_utils import apply_filter

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from error_detector_for_plotting import *

# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}

unique_str = str(time.time())  # prevent overlaps when running two scripts
assert not exists("tmp_{}.png".format(unique_str))

def playback_trajectory_with_env(
    env, 
    initial_state, 
    states,
    obs,
    detector,
    error_state=None,
    actions=None,
    video_writer=None,
    camera_names=None,
):
    assert isinstance(env, EnvBase)

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    traj_len = states.shape[0]
    
    #traj_len = 20

    if actions is not None:
        assert states.shape[0] == actions.shape[0]

    # Obtain images from states
    video_lst = []
    for i in range(traj_len):
        time.sleep(0.05)
        env.reset_to({"states": states[i]})
        video_img = []
        for cam_name in camera_names:
            video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
        video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
        video_lst.append(video_img)



    if error_state is None:
        if actions is not None:
            plot_data = detector.evaluate_trajectory(actions, obs)
        else:
            # Obtain error plot from error detector
            plot_data = detector.evaluate_trajectory(obs)
    else:
        if isinstance(detector, BCDreamer_Failure):
            error_label, error_prob = error_state
            plot_data_label = detector.evaluate_trajectory(error_label)
            plot_data_prob = detector.evaluate_trajectory(error_prob, use_prob_eval=True)

            # also plot the single future curve to compare the reduction in noise
            if len(plot_data_prob.shape) == 3:
                plot2 = plot_data_prob[:, 0, 1]
            else:
                plot2 = plot_data_prob[:, 0, 0, 1]

            plot_data_prob_std = plot_data_prob.std(axis=tuple(range(len(plot_data_prob.shape)))[1:-1])[:, 1]
            plot_data_prob = plot_data_prob.mean(axis=tuple(range(len(plot_data_prob.shape)))[1:-1])[:, 1]
        else:
            error_current, error_imagine = error_state
            plot_data_current = detector.evaluate_trajectory(error_current)
            plot_data_imagine = detector.evaluate_trajectory(error_imagine, use_img_eval=True)

    # Obtain plot images
    plot_images = []
    fig, ax = plt.subplots()
    plt.rcParams['figure.dpi'] = 128
    plt.figure(figsize=(5, 4))

    if isinstance(detector, BCDreamer_Failure):
        ymax = 1
    
        for i in range(traj_len):
            plt.title(detector.metric(), fontsize=18)
            plt.xlabel("Timesteps (t)", fontsize=15)
            plt.ylabel(detector.metric(), fontsize=15)

            plt.ylim(0, ymax)
            plt.xlim(0, detector.horizon())
            
            plt.plot(plot2[:i+1], label="prob_single_future", color='C0')
            plt.plot(plot_data_prob[:i+1], label="prob_multi_future", color='C1')
            plt.fill_between(np.arange(i+1), plot_data_prob[:i+1]-plot_data_prob_std[:i+1], plot_data_prob[:i+1]+plot_data_prob_std[:i+1], color='C1', alpha=0.1)
            
            plt.savefig("tmp_{}.png".format(unique_str))
            plt.clf()
            plot_image = cv2.imread("tmp_{}.png".format(unique_str))
            plot_images.append(plot_image)

        # Write video with image filters and plot images
        for i in range(len(video_lst)):
            video_img = video_lst[i]
            plot_image = plot_images[i]
            # is_preintv = plot_data_label[i] == 1 #detector.above_threshold(plot_data[i])
            # is_intv = plot_data_label[i] == 2
            # if is_preintv:
            #     video_img = apply_filter(video_img, color=(255, 0, 0))
            # elif is_intv:
            #     video_img = apply_filter(video_img, color=(0, 255, 0))
            video_img = np.concatenate([video_img, plot_image], axis=1) # concatenate horizontally
            video_writer.append_data(video_img)
            
    elif isinstance(detector, BCDreamer_SVM):
        ymax = 1500 #max(500, np.max(plot_data_current))
        ymin = -1500 #min(-500, np.min(plot_data_current))
        
        for i in range(traj_len):
            ax.set_xlabel("Timesteps (t)", fontsize=15)
            ax.set_title(detector.metric(), fontsize=18)
            plt.ylim(ymin, ymax)
            plt.xlim(0, detector.horizon())
            plt.plot(range(i+1), plot_data_current[:i+1], label="current")
            plt.plot(range(i+1), plot_data_imagine[:i+1], label="imagine")
            plt.legend(loc='lower right')
            plt.savefig("tmp_{}.png".format(unique_str))
            plt.clf()
            plot_image = cv2.imread("tmp_{}.png".format(unique_str))
            plot_images.append(plot_image)


        # Write video with image filters and plot images
        for i in range(len(video_lst)):
            video_img = video_lst[i]
            plot_image = plot_images[i]
            is_novel = detector.above_threshold(plot_data_current[i])
            if is_novel:
                video_img = apply_filter(video_img, color=(255, 0, 0))
            video_img = np.concatenate([video_img, plot_image], axis=1) # concatenate horizontally
            video_writer.append_data(video_img)
           
    else:
        ymax = 0.2
        
        print(plot_data_current[:traj_len])
        print(plot_data_imagine[:traj_len])

        for i in range(traj_len):
            ax.set_xlabel("Timesteps (t)", fontsize=15)
            ax.set_title(detector.metric(), fontsize=18)
            plt.ylim(0, ymax)
            plt.xlim(0, detector.horizon())
            plt.plot(range(i+1), plot_data_current[:i+1], label="current")
            plt.plot(range(i+1), plot_data_imagine[:i+1], label="imagine")
            plt.legend(loc='lower right')
            plt.savefig("tmp_{}.png".format(unique_str))
            plt.clf()
            plot_image = cv2.imread("tmp_{}.png".format(unique_str))
            plot_images.append(plot_image)


        # Write video with image filters and plot images
        for i in range(len(video_lst)):
            video_img = video_lst[i]
            plot_image = plot_images[i]
            is_novel = detector.above_threshold(plot_data_current[i])
            if is_novel:
                video_img = apply_filter(video_img, color=(255, 0, 0))
            video_img = np.concatenate([video_img, plot_image], axis=1) # concatenate horizontally
            video_writer.append_data(video_img)
            
def playback_dataset(args):
    # some arg checking
    write_video = (args.video_path is not None)

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    # create environment only if not playing back with observations
    if not args.use_obs:
        # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
        # for playback since observations are unused. Pass a dummy spec here.
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)

        # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

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
        demos = demos[:args.n]

    video_writer = imageio.get_writer(args.video_path, fps=20)

    if args.detector_type == "momart":
        detector = VAEMoMaRT(args.detector_checkpoints[0], 0.05)
    if args.detector_type == "vae_action":
        detector = VAE_Action(args.detector_checkpoints[0], 0.05)
    elif args.detector_type == "ensemble":
        assert len(args.detector_checkpoints) >= 3
        detector = Ensemble(args.detector_checkpoints, 0.05)
    elif args.detector_type == "bc_dreamer_ood":
        detector = BCDreamer_OOD(args.detector_checkpoints[0], 
                                 threshold=args.threshold, 
                                 demos_embedding_path=args.demos_embedding_path
                                 )
    elif args.detector_type == "bc_dreamer_failure":
        detector = BCDreamer_Failure(args.detector_checkpoints[0], 
                                     threshold=args.threshold, 
                                     use_prob=args.use_prob,
                                     )
    elif args.detector_type == "bc_dreamer_svm":
        detector = BCDreamer_SVM(args.detector_checkpoints[0], 
                                     threshold=args.threshold, 
                                     demos_embedding_path=args.demos_embedding_path
                                     )

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        
        try:
            obs = f["data/{}/obs".format(ep)]
        except:
            obs = None

        # supply actions if using open-loop action playback
        actions = None
        if args.detector_type == "vae_action":
            actions = f["data/{}/actions".format(ep)][()]
            
        if args.detector_type == "bc_dreamer_ood":
            if not args.use_imagine:    
                error_state = np.load(args.error_state_dir + "/latent_{}.npy".format(ind))
                error_state_current = np.squeeze(error_state, 1)
                
                error_state = np.load(args.error_state_dir + "/img_embedding_{}.npy".format(ind))
                error_state = error_state[:,4,:]
                error_state_imagine = np.squeeze(error_state, 1)
                
                assert error_state_current.shape[0] == states.shape[0] 
                assert error_state_imagine.shape[0] == states.shape[0] 
                error_state = (error_state_current, error_state_imagine)
                
            else:
                error_state = np.load(args.error_state_dir + "/img_embedding_{}.npy".format(ind))
                error_state = error_state[:,4,:]
                error_state = np.squeeze(error_state, 1)
                assert error_state.shape[0] == states.shape[0] # confirm correct error_state
                
        elif args.detector_type == "bc_dreamer_failure":
            if not args.use_prob:
                error_label = np.load(args.error_state_dir + "/rewards_{}.npy".format(ind), allow_pickle=True)
                error_probs = np.load(args.error_state_dir + "/rewards_probs{}.npy".format(ind), allow_pickle=True)
                error_state = (error_label, error_probs)

                print(error_label.shape, error_probs.shape, states.shape)
                assert error_label.shape[0] == states.shape[0] # confirm correct error_state
                assert error_probs.shape[0] == states.shape[0] # confirm correct error_state
            else:
                error_state = np.load(args.error_state_dir + "/rewards_probs{}.npy".format(ind))
                assert error_state.shape[0] == states.shape[0] # confirm correct error_state

        elif args.detector_type == "bc_dreamer_svm":
            if not args.use_imagine:    
                error_state = np.load(args.error_state_dir + "/latent_{}.npy".format(ind))
                error_state_current = np.squeeze(error_state, 1)
                
                error_state = np.load(args.error_state_dir + "/img_embedding_{}.npy".format(ind))
                error_state = error_state[:,-1,:]
                
                error_state_imagine = error_state
                #error_state_imagine = np.squeeze(error_state, 1)
                
                assert error_state_current.shape[0] == states.shape[0] 
                assert error_state_imagine.shape[0] == states.shape[0] 
                error_state = (error_state_current, error_state_imagine)
                
            else:
                error_state = np.load(args.error_state_dir + "/img_embedding_{}.npy".format(ind))
                error_state = error_state[:,4,:]
                error_state = np.squeeze(error_state, 1)
                assert error_state.shape[0] == states.shape[0] # confirm correct error_state

        playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state,
            states=states,
            obs=obs,
            detector=detector,
            actions=actions,
            error_state=error_state,
            video_writer=video_writer,
            camera_names=args.render_image_names,
        )

    f.close()
    if write_video:
        video_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
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

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
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

    parser.add_argument(
        "--detector_type",
        type=str,
        choices=["momart", "ensemble", "vae_action", 
                 "bc_dreamer_ood", "bc_dreamer_failure", "bc_dreamer_svm"
                 ]
    )

    parser.add_argument(
        "--detector_checkpoints",
        type=str,
        nargs='+',
    )

    # for bc_dreamer_ood
    parser.add_argument(
        "--demos_embedding_path",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--use_imagine",
        action="store_true",
    )

    # for bc_dreamer_failure
    parser.add_argument(
        "--use_prob",
        action='store_true',
    )

    # for both bc_dreamer_ood and bc_dreamer_failure
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--error_state_dir",
        type=str,
        default=None,
    )
    

    args = parser.parse_args()
    playback_dataset(args)
