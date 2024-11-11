"""Use meta policy mechanism for running the policy."""
from torch.distributions import MixtureSameFamily
import os
import argparse
import h5py
import random
import ast

from deoxys.franka_interface import FrankaInterface
# from deoxys.k4a_interface import K4aInterface
from rpl_vision_utils.networking.camera_redis_interface import CameraRedisSubInterface
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.input_utils import input2action

from deoxys.utils import YamlConfig
import torch

import init_path
from utils import resize_img, safe_cuda
from policy import VanillaBCPolicy
from hdf5_to_npy_image_real_robot import mat2quat

from PIL import Image, ImageDraw
from matplotlib import cm
import numpy  as np
import cv2
import imageio
from easydict import EasyDict
import json
import time

import hydra
from omegaconf import OmegaConf, DictConfig
import yaml
from easydict import EasyDict
from hydra.experimental import compose, initialize
import pprint
from pathlib import Path

from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.input_utils import input2action
from deoxys import config_root

from rpl_vision_utils.utils import img_utils as ImgUtils

from robomimic.utils.file_utils import policy_from_checkpoint
from error_detection_utils import detector_from_config

DEMO = -1
ROLLOUT = 0
INTV = 1

@hydra.main(config_path="./conf", config_name="config_hitl_ed")
def main(hydra_cfg):
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)

    robot_interface = FrankaInterface(os.path.join(config_root, cfg.robot.interface_cfg))

    assert cfg.env in ["gear", "kcup", "cleanup"]

    if cfg.env == "gear":
        controller_type = 'OSC_POSITION'
        controller_cfg = "osc-position-controller.yml"
    elif cfg.env == "kcup":
        controller_type = 'OSC_YAW'
        controller_cfg = "osc-yaw-controller.yml"
    elif cfg.env == "cleanup":
        controller_type = 'OSC_YAW'
        controller_cfg = "osc-yaw-controller.yml"
    else:
        raise NotImplementedError

    controller_cfg = YamlConfig(os.path.join(config_root, controller_cfg)).as_easydict()
    
    """ Data Collection Saving """
    from pathlib import Path
    #ckpt_num = cfg.checkpoint_dir[-7:-4]
    #folder = Path(cfg.save_folder + ckpt_num)
    folder = Path(cfg.save_folder)
    folder.mkdir(parents=True, exist_ok=True)

    experiment_id = 0
    for path in folder.glob('run*'): 
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split('run')[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    experiment_id += 1
    folder = str(folder / f"run{experiment_id}")
    os.makedirs(folder, exist_ok=True)

    """ End Data Collection Saving """

    """ Create data for saving """

    data = {"action": [],
            "proprio_ee": [],
            "proprio_joints": [],
            "proprio_gripper_state": [],
            "action_modes": [],
            "error_detected": []
            }
    
    camera_ids = [0, 1]    
    for camera_id in camera_ids:
        data[f"camera_{camera_id}"] = []

    demos_subtask_sequence = {}

    # """ Loading Robomimic Model """
    # eval_policy = policy_from_checkpoint(ckpt_path=cfg.checkpoint_dir)[0]
    # #print(eval_policy.policy)
    # eval_policy.policy.low_noise_eval = False
    # eval_policy.start_episode()
    # """ ======================= """

    """ Loading Error Detectors"""
    import torch
    if cfg.detector_type == 'thrifty':
        detector = detector_from_config(cfg.detector_type, cfg.detector_checkpoints)
        eval_policy = detector
    else:
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        detector = detector_from_config(cfg.detector_type, cfg.detector_checkpoints)
        detector.rollout_policy.policy.nets.to('cuda')
        
        eval_policy = detector.rollout_policy
        eval_policy.policy.low_noise_eval = False
        eval_policy.start_episode()

    print("==> Policy and Error Detector Loaded")
    """ ======================= """

    if cfg.env == "kcup":
        MAX_HORIZON = 3000
    elif cfg.env == "gear":
        MAX_HORIZON = 3000
    elif cfg.env == "cleanup":
        MAX_HORIZON = 1200

    seq = []
    camera_ids = [0, 1]

    cr_interfaces = {}
    for camera_id in camera_ids:
        cr_interface = CameraRedisSubInterface(camera_id=camera_id)
        cr_interface.start()
        cr_interfaces[camera_id] = cr_interface

    device = SpaceMouse(vendor_id=9583, product_id=50734)
    device.start_control()
    recorded_imgs = []
    video_dir = None

    starting_img_info = {}
    ending_img_info = {}
    for camera_id in camera_ids:
        starting_img_info[f"camera_{camera_id}"] = cr_interfaces[camera_id].get_img_info()

    gripper_history = []

    # while len(robot_interface._gripper_state_buffer) == 0 or len(robot_interface._state_buffer) == 0:
    #     import time; time.sleep(0.05)

    last_obs_dict = None
    obs_buffer = []

    gripper_state = -1
    prev_toggle = -1
    cur_toggle = -1
    for _ in range(MAX_HORIZON):
        start_time = time.time_ns()
        spacemouse_action, grasp = input2action(
            device=device,
            #control_type=controller_type
        )

        if spacemouse_action is None:
            break

        # Confirm the same with collect data script!
        spacemouse_action[:3] *= 1.0

        obs_images = []

        agentview_image = None
        eye_in_hand_image = None

        for camera_id in camera_ids:
            imgs = cr_interfaces[camera_id].get_img()
            img = imgs["color"]
            #img = cv2.imread(color_img)
            camera_type = "k4a" if camera_id == 0 else "rs"

            if cfg.env == "cleanup":
                fx_fy_dict = {0: {"fx": 0.20, "fy": 0.20},
                              1: {"fx": 0.2, "fy": 0.3}}
            elif cfg.env == "gear":
                fx_fy_dict = {0: {"fx": 0.27, "fy": 0.27},
                              1: {"fx": 0.2, "fy": 0.3}}
            else:
                fx_fy_dict = {0: {"fx": 0.20, "fy": 0.20},
                              1: {"fx": 0.2, "fy": 0.3}}

            if cfg.env == "cleanup":
                if camera_type == "k4a":
                    offset_w = 0
                    offset_h = 160
                    img_w = 128
                    img_h = 170
                elif camera_type == "rs":
                    offset_w = 0
                    offset_h = 0
                    img_w = 128
                    img_h = 128
            elif cfg.env == "gear":
                if camera_type == "k4a":
                    offset_w = 10
                    offset_h = 0
                elif camera_type == "rs":
                    offset_w = 0
                    offset_h = 0
            else:
                if camera_type == "k4a":
                    offset_w = 220
                    offset_h = 260
                    img_w = 500
                    img_h = 600

                elif camera_type == "rs":
                    offset_w = 0
                    offset_h = 0
                    img_w = img.shape[0]
                    img_h = img.shape[1]

            resized_img = img[offset_w: offset_w+img_w, offset_h:offset_h+img_h, :]
            
            # print(camera_type, resized_img.shape)
            (_w, _h, _)= resized_img.shape
            resized_color_img = cv2.resize(resized_img, (84,84))#, fx=128 / _w, fy=128 / _h)
            # print(resized_color_img.shape)

            if camera_id == 0:
                agentview_image = resized_color_img
                obs_images.append(agentview_image)
                # if cfg.eval.video:
                #     recorded_imgs.append(color_img)
            elif camera_id == 1:
                eye_in_hand_image = resized_color_img
                obs_images.append(eye_in_hand_image)
            else:
                raise NotImplementedError

           
        #     if camera_id == 0:
        #         cv2.namedWindow("agentview")
        #         cv2.moveWindow("agentview", 0, 500)
        #         cv2.imshow("agentview", resized_color_img)
        #         #cv2.imshow("agentview", resize_img(color_img, cr_interfaces[camera_id].camera_type))


        #     if camera_id == 1:
        #         cv2.imshow("eye_in_hand", resized_color_img)
        #         #cv2.imshow("eye_in_hand", resize_img(color_img, cr_interfaces[camera_id].camera_type))
          
        # cv2.waitKey(1)
           
        """ Creating Robomimic Observational Space """
        ee_proprio_orig = np.array(robot_interface._state_buffer[-1].O_T_EE)
        
        ee_pos = ee_proprio_orig[12:15]
        
        ee_proprio = ee_proprio_orig.reshape((4, 4))
        rot_matrix = ee_proprio[:3, :3]
        ee_quat = mat2quat(rot_matrix)

        agentview_image = np.transpose(obs_images[0], (2, 0, 1))
        agentview_image = np.float32(agentview_image) / 255.0
        
        eih_image = np.transpose(obs_images[1], (2, 0, 1))
        eih_image = np.float32(eih_image) / 255.0

        last_gripper_state = robot_interface._gripper_state_buffer[-1]

        last_state = robot_interface._state_buffer[-1]
        joint_states = np.array(last_state.q)

        gripper_states = np.array([last_gripper_state.width])

        obs = {
                'agentview_image': agentview_image, 
                'eye_in_hand_image': eih_image,
                'ee_states': ee_proprio_orig,
                'gripper_states': gripper_states,
                'joint_states': joint_states,
                # 'robot0_gripper_qpos': np.array([gripper_state]),
                # 'robot0_eef_quat': ee_quat,
                # 'robot0_eef_pos': ee_pos,
                }

        for state in ["ee_states", "joint_states", "gripper_states"]:
            if np.sum(np.abs(obs[state])) <= 1e-6 and last_obs_dict is not None:
                print(f"{state} missing!!!!")
                obs[state] = last_obs_dict[state]
        
        policy_action = eval_policy(obs)
        obs_buffer.append(obs)
        human_intervene = detector.human_intervene(obs_buffer)
        if human_intervene:
            print("Intervene!!", time.time())        

        """ Either Use Spacemouse Action or Policy Action """
        if  np.linalg.norm(spacemouse_action) != 1.0:
            # if  human_intervene or np.linalg.norm(spacemouse_action) != 1.0:
            action = spacemouse_action
            action_mode = INTV
            
            prev_toggle = cur_toggle
            cur_toggle = spacemouse_action[-1]
            if prev_toggle == -1 and cur_toggle == 1:
                gripper_state *= -1
            action[-1] = gripper_state
            # print('human', action)
        else:
            action = policy_action
            gripper_state = action[-1]
            action_mode = ROLLOUT
            # print('policy', action)
        """ ====================================== """

        if len(action) == 5:
            b_action = np.append(action[:3], [0,0])
            action = np.append(b_action, [action[-2], action[-1]])
        if len(action) == 4:
            b_action = np.append(action[:3], [0,0,0])
            action = np.append(b_action, [action[-1]])
        
        robot_interface.control(controller_type=controller_type,
                                action=action,
                                controller_cfg=controller_cfg)

        """ Saving Data at the End """
        last_state = robot_interface._state_buffer[-1]

        # print(np.round(np.array(last_state.q), 3))
        data["action"].append(action)
        data["proprio_ee"].append(np.array(last_state.O_T_EE))
        data["proprio_joints"].append(np.array(last_state.q))
        data["proprio_gripper_state"].append(np.array(gripper_states))
        data["action_modes"].append(np.array(action_mode))
        data["error_detected"].append(np.array(int(human_intervene)))
        # Get img info

        for camera_id in camera_ids:
            img_info = cr_interfaces[camera_id].get_img_info()
            data[f"camera_{camera_id}"].append(img_info)

        last_obs_dict = obs

        end_time = time.time_ns()
        # print(f"Time profile: {(end_time - start_time) / 10 ** 9}")

    gripper_data = np.array(data["proprio_gripper_state"])
    gripper_data = np.squeeze(gripper_data, axis=1)

    np.savez(f"{folder}/testing_demo_action", data=np.array(data["action"]))
    np.savez(f"{folder}/testing_demo_ee_states", data=np.array(data["proprio_ee"]))
    np.savez(f"{folder}/testing_demo_joint_states", data=np.array(data["proprio_joints"]))
    np.savez(f"{folder}/testing_demo_gripper_states", data=gripper_data)
    np.savez(f"{folder}/testing_demo_action_modes", data=np.array(data["action_modes"]))
    np.savez(f"{folder}/testing_demo_error_detected", data=np.array(data["error_detected"]))

    assert len(data["action"]) == len(data["proprio_ee"]) == len(data["proprio_joints"]) == len(data["proprio_gripper_state"]) \
            == len(data["action_modes"])

    for camera_id in camera_ids:
        np.savez(f"{folder}/testing_demo_camera_{camera_id}", data=np.array(data[f"camera_{camera_id}"]))
        cr_interfaces[camera_id].stop()

    robot_interface.close()

    # Saving
    valid_input = False
    while not valid_input:
        try:
            save = input("Save or not? (enter 0 or 1)")
            save = bool(int(save))
            valid_input = True
        except:
            pass
   
    if not save:
        import shutil
        shutil.rmtree(f"{folder}")
        exit()

    if not cfg.intv:
        # Record success for eval mode
        valid_input = False
        while not valid_input:
            try:
                success = input("Success or fail? (enter 0 or 1)")
                success = bool(int(success))
                valid_input = True
            except:
                pass

        if success:
            np.savez(f"{folder}/success", data=np.array(success))

    # Printing dataset info
    import subprocess
    subprocess.run([
        'python', '/home/huihanl/robot_control_ws/robot_infra/gprs/examples/count_intv.py',
        '--folder', cfg.save_folder,
        '--env', cfg.env,
    ])

if __name__ == "__main__":
    main()
