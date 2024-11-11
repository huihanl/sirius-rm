import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robosuite import load_controller_config
import robosuite
from robosuite.wrappers import VisualizationWrapper
from robosuite.wrappers import DataCollectionWrapper
import robomimic.utils.tensor_utils as TensorUtils
import numpy as np
import copy
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import numpy as np
import cv2
import imageio


np.random.seed(0)
torch.manual_seed(0)

ckpt_path = '/home/shivin/hitl/experiments/square/thrifty/05-19-hitl-shivin/gamma0.995_lr0.0001_policy-acts/2023-05-21-21-07-08/models/model_epoch_100.pth'
ckpt_path = '/home/shivin/hitl/experiments/square/thrifty/05-19-hitl-shivin/gamma0.999_lr0.0001_policy-act/2023-05-21-22-02-56/models/model_epoch_200.pth'
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path)
policy = policy.policy

# config = {
#     "env_name": "NutAssemblySquare",
#     "robots": "Panda",
#     "controller_configs": load_controller_config(default_controller="OSC_POSE"),
# }

# env =  robosuite.make(
#         **config,
#         has_renderer=False,
#         has_offscreen_renderer=True,
#         render_camera="agentview",
#         ignore_done=True,
#         use_camera_obs=True,
#         reward_shaping=False,
#         control_freq=20,
#     )

# env = DataCollectionWrapper(VisualizationWrapper(env),  "/tmp/{}".format(str(time.time()).replace(".", "_")))

env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=None,#args.env, 
        render=False,#args.render, 
        render_offscreen=False,#(args.video_path is not None), 
        verbose=True,
    )


plt.figure(figsize=(5, 4))
video_writer = imageio.get_writer("/home/shivin/hitl/experiments/square/thrifty_rollouts/vid3.mp4", fps=20)

unique_str = str(time.time()).replace(".", "_")
for i in range(5):
    obs = env.reset()
    q_vals = []
    for j in tqdm(range(300)):
        vid_img = env.render(mode="rgb_array", height=512, width=512, camera_name="agentview")

        obs = TensorUtils.to_tensor(obs)
        obs = TensorUtils.to_batch(obs)
        obs = TensorUtils.to_device(obs, policy.device)
        obs = TensorUtils.to_float(obs)

        action = policy.get_single_step_action(obs)

        q_val = policy.get_q_safety(obs, action)
        q_vals.append(q_val)

        obs, _, done, _ = env.step(action[0].cpu().numpy())

        if done or env.is_success()["task"]:
            print("Success!")
            break

        # if True:
        #     continue

        if j%5 ==0:
            # create vid of env and plot
            plt.title("q_val", fontsize=18)
            plt.xlabel("Timesteps (t)", fontsize=15)
            plt.ylabel("q_val", fontsize=15)

            plt.ylim(0, 1.1)
            plt.xlim(0, 300)
            
            plt.plot(q_vals, label="q-predicted", color='C1')
            
            plt.savefig("tmp_{}.png".format(unique_str))
            plt.clf()
            plot_image = cv2.resize(cv2.imread("tmp_{}.png".format(unique_str)), (512, 512))
            # plot_images.append(plot_image)
        
            video_writer.append_data(np.concatenate([vid_img, plot_image], axis=1))

video_writer.close()