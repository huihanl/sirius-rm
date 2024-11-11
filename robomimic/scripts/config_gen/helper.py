import argparse
import os
import robomimic
import time
import datetime

base_path = os.path.abspath(os.path.join(os.path.dirname(robomimic.__file__), os.pardir))
from robomimic.scripts.config_gen.data_path import data_base_path
from robomimic.scripts.config_gen.dataset_names_def import *


def set_filter_key(generator, args):
    if args.filter_key is None:
        return

    """ Specific to rounds """
    if "round" in args.filter_key:
        round_nums = args.filter_key[5:]
        key_list = ["round" + n for n in round_nums]
        print(key_list)
        generator.add_param(
            key="train.hdf5_filter_key",
            name="hdf5_filter",
            group=-9999,
            values=[
                key_list
            ],
            value_names=[
                args.filter_key,
            ]
        )
        return

    generator.add_param(
        key="train.hdf5_filter_key",
        name="hdf5_filter",
        group=-9999,
        values=[args.filter_key],
    )


def set_warmstart(generator, args):
    if args.warmstart < 0:
        return
    generator.add_param(
        key="experiment.rollout.warmstart",
        name="warmstart",
        group=-9999,
        values=[args.warmstart],
        hidename=True,
    )


def no_resmlp(generator, args, algo):
    if not args.no_resmlp:
        return

    if algo == "awac":
        generator.add_param(
            key="algo.actor.res_mlp.enabled",
            name="resmlp",
            group=793,
            values=[False],
            value_names=["F"],
            hidename=True,
        )
    elif algo == "bc":
        generator.add_param(
            key="algo.res_mlp.enabled",
            name="resmlp",
            group=793,
            values=[False],
            value_names=["F"],
            hidename=True,
        )
    else:
        assert NotImplementedError


def set_resmlp(generator, args, algo):
    if not args.sweep_resmlp:
        return

    if algo == "awac":
        generator.add_param(
            key="algo.actor.res_mlp.enabled",
            name="resmlp",
            group=793,
            values=[True, False],
            value_names=["T", "F"]
        )
    elif algo == "bc":
        generator.add_param(
            key="algo.res_mlp.enabled",
            name="resmlp",
            group=793,
            values=[True, False],
            value_names=["T", "F"]
        )
    else:
        assert NotImplementedError


def set_sweep_iwr(generator, args):
    if not args.sweep_iwr:
        return

    generator.add_param(
        key="train.use_iwr_sampling",
        name="",
        group=-1234,
        values=[True, False],
        value_names=["T", "F"]
    )

def set_iwr(generator, args):
    if args.iwr:
        generator.add_param(
            key="train.use_iwr_sampling",
            name="iwr_sampling",
            group=-1234,
            values=[True],
            value_names=["T"],
        )
    else:
        generator.add_param(
            key="train.use_iwr_sampling",
            name="iwr_sampling",
            group=-1234,
            values=[False],
            value_names=["F"],
            hidename=True,
        )

def set_bootstrap(generator, args):
    if args.bootstrap:
        generator.add_param(
            key="train.bootstrap_sampling",
            name="bs_sampling",
            group=-1234,
            values=[True],
            value_names=["T"],
        )
    else:
        generator.add_param(
            key="train.bootstrap_sampling",
            name="",
            group=-1234,
            values=[False],
            hidename=True,
        )

def set_img_size_128(generator, args):
    if not args.img_size128:
        return
    print("DEBUG: USING 116 CROP SIZE")
    generator.add_param(
        key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
        name="crop",
        group=-1,
        values=[116],
    )
    generator.add_param(
        key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
        name="",
        group=-1,
        values=[116],
    )


def set_img_size_240(generator, args):
    if not args.img_size240:
        return
    print("DEBUG: USING 216 CROP SIZE")
    generator.add_param(
        key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
        name="",
        group=-1,
        values=[216],
    )
    generator.add_param(
        key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
        name="",
        group=-1,
        values=[216],
    )


def set_large_scale_eval_mode(generator, args):
    if not args.large_scale:
        return

    generator.add_param(
        key="experiment.rollout.n",
        name="rollout_n",
        group=-99,
        values=[200],
        hidename=True,
    )

    # train.seed
    generator.add_param(
        key="train.seed",
        name="seed",
        group=-98,
        values=[1, 2, 3],
    )


def set_mid_scale_eval_mode(generator, args):
    if not args.mid_scale:
        return

    generator.add_param(
        key="experiment.rollout.n",
        name="rollout_n",
        group=-99,
        values=[100],
        hidename=True,
    )

    # train.seed
    generator.add_param(
        key="train.seed",
        name="seed",
        group=-98,
        values=[1, 2],
    )


def set_num_epochs(generator, args):
    if args.num_epoches < 0:
        return
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[args.num_epoches],
        value_names=[""],
        hidename=True,
    )


def set_debug_mode(generator, args):
    if not args.debug:
        return

    generator.add_param(
        key="experiment.rollout.n",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.rollout.horizon",
        name="",
        group=-1,
        values=[100],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.rollout.rate",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.epoch_every_n_steps",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.keep_all_videos",
        name="",
        group=-1,
        values=[True],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.validation_epoch_every_n_steps",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.save.enabled",
        name="",
        group=-1,
        values=[False],
        value_names=[""],
    )
    generator.add_param(
        key="train.hdf5_cache_mode",
        name="",
        group=-1,
        values=[None],
        value_names=[""],
    )


def set_exp_id(generator, args):
    assert args.name is not None

    vals = generator.parameters["train.output_dir"].values

    for i in range(len(vals)):
        vals[i] = os.path.join(vals[i], args.name)

    if (args.debug or args.tmplog) and (args.wandb_proj_name is None):
        args.wandb_proj_name = 'debug'


def set_num_rollouts(generator, args):
    if args.nr < 0:
        return
    generator.add_param(
        key="experiment.rollout.n",
        name="",
        group=-1,
        values=[args.nr],
        hidename=True,
    )


def set_rollout_rate(generator, args):
    if args.rollout_rate < 0:
        return

    generator.add_param(
        key="experiment.rollout.rate",
        name="",
        group=-1,
        values=[80],
        value_names=[""],
        hidename=True,
    )


def set_num_seeds(generator, args):
    if args.ns < 0:
        return

    seeds = []
    for i in range(args.ns):
        seeds.append(1 + i)

    print("Using seeds: ")
    print(seeds)

    generator.add_param(
        key="train.seed",
        name="seed",
        group=-98,
        values=seeds,
    )


def set_iwr_and_ft_settings(generator, args):
    if args.ft:
        assert args.ckpt_path is not None
        generator.add_param(
            key="experiment.ckpt_path",
            name="ft",
            group=-1,
            values=[args.ckpt_path],
            value_names=["yes"],
        )

def set_dataset(generator, args):
    if args.dataset is not None:
        generator.add_param(
            key="train.data",
            name="",
            group=-1,
            values=[args.dataset],
            value_names=[""],
        )


def set_wandb_mode(generator, args):
    if args.no_wandb:
        generator.add_param(
            key="experiment.logging.log_wandb",
            name="",
            group=-1,
            values=[False],
        )


def set_gradnorm(generator, args, algo):
    if args.gradnorm < 0:
        return
    if algo == "bc":
        generator.add_param(
            key="algo.max_gradient_norm",
            name="gradnorm",
            group=-1,
            values=[args.gradnorm],
        )
    elif algo == "awac":
        generator.add_param(
            key="algo.actor.max_gradient_norm",
            name="gradnorm",
            group=-1,
            values=[args.gradnorm],
        )
    else:
        raise ValueError


def set_no_eval(generator, args):
    if not args.no_eval:
        return
    generator.add_param(
        key="experiment.rollout.enabled",
        name="",
        group=-1,
        values=[False],
    )


def set_learning_rate(generator, args):
    if args.lr < 0:
        return
    generator.add_param(
        key="algo.optim_params.policy.learning_rate.initial",
        name="lr",
        group=-1,
        values=[args.lr],
    )


def set_weight_decay(generator, args):
    if args.wd < 0:
        return
    generator.add_param(
        key="algo.optim_params.policy.regularization.L2",
        name="wd",
        group=-1,
        values=[args.wd],
    )

reweight_dict_list = [
        #{1 : 0.8, 2 : 0.9, 3: 1.1, 4: 1.2},
        {1 : 0.9, 2 : 1.0, 3: 1.1, },
        {1 : 0.8, 2 : 1.0, 3: 1.2, },
        {1 : 1, 2 : 1.2, 3: 1.5, },
        ]

def set_reweight_rounds(generator, args):
    if not args.reweight_rounds:
        return
    generator.add_param(
        key="algo.hc_weights.diff_weights_diff_rounds",
        name="rounds_reweight",
        group=-1067,
        values=[True],
        value_names=["T"],
    )
    generator.add_param(
        key="algo.hc_weights.round_upweights",
        name="rweight",
        group=-1069,
        values=reweight_dict_list,
        value_names=[0,1,2] #,1,2,3,4,5,6]
    )
    generator.add_param(
        key="algo.hc_weights.normalize_after_round_upweight",
        name="",
        group=-1069,
        values=[True] * 3, #[True] * 4 + [False] * 4,
        value_names=["T"] * 3, #["T"] * 4 + ["F"] * 4
    )

    generator.add_param(
        key="train.dataset_keys",
        name="",
        group=3,
        values=[
            (
                "actions",
                "rewards",
                "dones",
                "action_modes",
                "intv_labels",
                "round",
            )
        ],
    )


def set_round_resampling(generator, args):
    if not args.round_resample:
        return

    generator.add_param(
        key="algo.hc_weights.rounds_resampling",
        name="rounds_resmpl",
        group=-1067,
        values=[True],
        value_names=["T"],
    )
    generator.add_param(
        key="algo.hc_weights.resampling_weight",
        name="resmpl_weight",
        group=-1068,
        values=[2, 5],
    )

    generator.add_param(
        key="train.dataset_keys",
        name="",
        group=3,
        values=[
            (
                "actions",
                "rewards",
                "dones",
                "action_modes",
                "intv_labels",
                "round",
            )
        ],
    )


def use_human_relabeled_weights(generator, args):
    if not args.use_human_relabeled:
        return

    generator.add_param(
        key="train.preintv_relabeling.base_key",
        name="base_key",
        group=1111,
        values=[
            "intervention"
        ]
    )

    generator.add_param(
        key="train.dataset_keys",
        name="",
        group=3,
        values=[
            (
                "actions",
                "rewards",
                "dones",
                "action_modes",
                "intv_labels",
                "intervention"
            )
        ],
    )

def set_full_pass(generator, args):
    if args.full_pass:
        generator.add_param(
            key="experiment.epoch_every_n_steps",
            name="full_pass",
            group=-30000,
            values=[
                None
            ],
        )

def set_horizon(generator, args):           
    if args.rollout_horizon < 0:
        return

    generator.add_param(
        key="experiment.rollout.horizon",
        name="",
        group=-1,
        values=[args.rollout_horizon],
    )

def set_use_sampler(generator, args):
    if not args.not_use_sampler:
        return

    generator.add_param(
        key="train.use_sampler",
        name="use_sampler",
        group=-3000000,
        values=[
            False,
        ],
        value_names=[
            "F",
        ]
    )

def use_weight_sampler(generator, args):
    if not args.use_weight_sampler:
        return

    generator.add_param(
        key="algo.hc_weights.use_weighted_sampler",
        name="w_sampler",
        group=-3000000,
        values=[
            True,
        ],
        value_names=[
            "T",
        ]
    )

def use_joint(generator, args):
    if not args.use_joint:
        return
    if 'real' in args.env:
        # deals with real seperately
        return

    generator.add_param(
        key="observation.modalities.obs.low_dim",
        name="ld",
        group=-1,
        values=[
            [
                "robot0_joint_pos",
                "robot0_gripper_qpos"
            ]
        ],
        value_names=["use_joint"],
        hidename=True,
    )

def set_no_validate(generator, args):
    if not args.no_val:
        return

    generator.add_param(
        key="experiment.validate",
        name="val",
        group=-1,
        values=[
            False,
        ],
        value_names=[
            "F",
        ]
    )


def set_no_intv(generator, args):
    if not args.no_intv:
        return

    generator.add_param(
        key="algo.remove_intervention_sampler",
        name="no_intv",
        group=-1,
        values=[
            True,
        ],
        value_names=[
            "T",
        ]
    )

def set_env_settings(generator, args):
    if 'real' in args.env:
        if args.modality == "im":
            if args.use_joint:
                generator.add_param(
                    key="observation.modalities.obs.low_dim",
                    name="ld",
                    group=-1,
                    values=[
                        ["joint_states",
                         "gripper_states"]
                    ],
                    value_names=["use_joint"]
                )
            else:
                generator.add_param(
                    key="observation.modalities.obs.low_dim",
                    name="ld",
                    group=-1,
                    values=[
                        ["ee_states",
                         "gripper_states"]
                    ],
                    value_names=["use_ee"]
                )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["agentview_image",
                     "eye_in_hand_image"]
                ],
            )
        elif args.modality == "ld":
            if args.use_joint:
                generator.add_param(
                    key="observation.modalities.obs.low_dim",
                    name="ld",
                    group=-1,
                    values=[
                        ["joint_states",
                         "gripper_states",
                         "objects"],
                    ],
                    value_names=["use_joint"]
                )
            else:
                generator.add_param(
                    key="observation.modalities.obs.low_dim",
                    name="ld",
                    group=-1,
                    values=[
                        ["ee_states",
                         "gripper_states",
                         "objects"],
                    ],
                    value_names=["use_ee"]
                )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    []
                ],
            )

        if args.img_size == 128:
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="",
                group=-1,
                values=[
                    116,  # for 128 image size
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[
                    116,  # for 128 image size
                ],
            )

        elif args.img_type == "113_84":
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="crop",
                group=-1,
                values=[76],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[102],
            )
            generator.add_param(
                key="observation.encoder.rgb2.obs_randomizer_kwargs.crop_height",
                name="crop2",
                group=-1,
                values=[76],
            )
            generator.add_param(
                key="observation.encoder.rgb2.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[76],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["agentview_image",
                    ]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb2",
                name="",
                group=-1,
                values=[
                    ["eye_in_hand_image",
                     ]
                ],
            )
        else: 
            # default img size 84
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="",
                group=-1,
                values=[
                    76,  # for 84 image size
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[
                    76,  # for 84 image size
                ],
            )

        generator.add_param(
            key="experiment.rollout.enabled",
            name="",
            group=-1,
            values=[False],
        )
        generator.add_param(
            key="train.dataset_keys",
            name="",
            group=-1,
            values=[
                (
                    "actions",
                    "action_modes",
                    "intv_labels"
                )
            ],
        )
        generator.add_param(
            key="experiment.save.every_n_epochs",
            name="",
            group=-1,
            values=[50],
        )

    elif args.env == 'tool_hang':
        generator.add_param(
            key="experiment.rollout.horizon",
            name="",
            group=-1,
            values=[700],
        )

        if args.img_type is None: # 128, 128
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="crop",
                group=-1,
                values=[116],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[116],
            )
        elif args.img_type == "birdview":
            
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    [
                        "birdview_image",
                        "robot0_eye_in_hand_image",
                    ]
                ],
            )

            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="crop",
                group=-1,
                values=[116],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[116],
            )

        elif args.img_type == "birdview_vae":

            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    [
                        "birdview_image",
                    ]
                ],
            )

            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="crop",
                group=-1,
                values=[116],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[116],
            )


        elif args.img_type == "240": # 240, 240
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="",
                group=-1,
                values=[
                    216,  # for 240 image size
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[
                    216,  # for 240 image size
                ],
            )
        elif args.img_type == "113_84": # 240, 84
            print("REACH HERE")
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="crop",
                group=-1,
                values=[102],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[76],
            )
            generator.add_param(
                key="observation.encoder.rgb2.obs_randomizer_kwargs.crop_height",
                name="crop2",
                group=-1,
                values=[76],
            )
            generator.add_param(
                key="observation.encoder.rgb2.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[76],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["agentview_image",
                    ]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb2",
                name="",
                group=-1,
                values=[
                    ["eye_in_hand_image",
                     ]
                ],
            )
        elif args.img_type == "180_84": # 180, 84
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="crop",
                group=-1,
                values=[162],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[162],
            )
            generator.add_param(
                key="observation.encoder.rgb2.obs_randomizer_kwargs.crop_height",
                name="crop2",
                group=-1,
                values=[76],
            )
            generator.add_param(
                key="observation.encoder.rgb2.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[76],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["agentview_image",
                    ]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb2",
                name="",
                group=-1,
                values=[
                    ["eye_in_hand_image",
                     ]
                ],
            )

    elif args.env == 'coffee_v0':
        if not args.debug:
            generator.add_param(
                key="experiment.rollout.horizon",
                name="",
                group=-1,
                values=[450],
            )
        if args.modality == "im":
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["agentview_image",
                     "robot0_eye_in_hand_image"]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    ]
                ],
            )
        elif args.modality == "ld":
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    "object"
                    ]
                ],
            )
            generator.add_param(
                key="experiment.render_video",
                name="",
                group=-1,
                values=[True],
            )

    elif args.env == "cleanup_real":
        if args.img_type == "113_84": # 180, 84
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="crop",
                group=-1,
                values=[102],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[76],
            )
            generator.add_param(
                key="observation.encoder.rgb2.obs_randomizer_kwargs.crop_height",
                name="crop2",
                group=-1,
                values=[76],
            )
            generator.add_param(
                key="observation.encoder.rgb2.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[76],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["agentview_image",
                    ]
                ],
            )
            generator.add_param(
                key="observation.modalities.obs.rgb2",
                name="",
                group=-1,
                values=[
                    ["eye_in_hand_image",
                     ]
                ],
            )

    elif args.env == 'threading_v0':
        if not args.debug:
            generator.add_param(
                key="experiment.rollout.horizon",
                name="",
                group=-1,
                values=[700],
            )
    elif args.env == 'door_v0':
        generator.add_param(
            key="experiment.rollout.horizon",
            name="",
            group=-10,
            values=[200],
        )
        generator.add_param(
            key="experiment.rollout.terminate_on_success",
            name="",
            group=-10,
            values=[False],
        )
        generator.add_param(
            key="observation.modalities.obs.low_dim",
            name="",
            group=-10,
            values=[["flat"]],
        )
        generator.add_param(
            key="experiment.render_video",
            name="",
            group=-10,
            values=[False],
        )
        generator.add_param(
            key="experiment.keep_all_videos",
            name="",
            group=-10,
            values=[False],
        )

def add_dataset(generator, args):
    if args.env == 'square':
        generator.add_param(
            key="train.data",
            name="ds",
            group=-30,
            values=[os.path.join(data_base_path, h) for h in dataset_hdf5s],
            value_names=dataset_names
        )
    elif args.env == 'tool_hang':
        generator.add_param(
            key="train.data",
            name="ds",
            group=-30,
            values=[os.path.join(data_base_path, h) for h in dataset_hdf5s_th],
            value_names=dataset_names_th
        )
    elif args.env == 'threading_v0':
        generator.add_param(
            key="train.data",
            name="ds",
            group=-30,
            values=[os.path.join(data_base_path, h) for h in dataset_hdf5s_thread],
            value_names=dataset_names_thread
        )
    elif args.env == 'coffee_v0':
        generator.add_param(
            key="train.data",
            name="ds",
            group=-30,
            values=[os.path.join(data_base_path, h) for h in dataset_hdf5s_coffee],
            value_names=dataset_names_coffee
        )
    elif args.env == 'gear_real':
        generator.add_param(
            key="train.data",
            name="ds",
            group=-30,
            values=[os.path.join(data_base_path, h) for h in dataset_hdf5s_gear],
            value_names=dataset_names_gear
        )
    elif args.env == 'kcup_real':
        generator.add_param(
            key="train.data",
            name="ds",
            group=-30,
            values=[os.path.join(data_base_path, h) for h in dataset_hdf5s_kcup],
            value_names=dataset_names_kcup
        )
    elif args.env == 'cleanup_real':
        generator.add_param(
            key="train.data",
            name="ds",
            group=-30,
            values=[os.path.join(data_base_path, h) for h in dataset_hdf5s_cleanup],
            value_names=dataset_names_cleanup
        )
    else:
        raise ValueError

def add_rnn(generator, args):

    if not args.rnn:
        return

    if args.seq_length <= 0:
        seq_length = 10
    else:
        seq_length = args.seq_length

    generator.add_param(
        key="train.seq_length",
        name="seql",
        group=-1,
        values=[seq_length],
        value_names=[seq_length],
    )
    generator.add_param(
        key="algo.actor_layer_dims",
        name="",
        group=-1,
        values=[[]],
        value_names=[""],
    )
    generator.add_param(
        key="algo.rnn.enabled",
        name="rnn",
        group=-1,
        values=[True],
        value_names=["T"],
    )
    generator.add_param(
        key="algo.rnn.horizon",
        name="rnn_h",
        group=-1,
        values=[seq_length],
        value_names=[seq_length],
    )

def set_no_rnn(generator, args):
    if not args.no_rnn:
        return 

    generator.add_param(
        key="train.seq_length",
        name="",
        group=-1,
        values=[1],
        value_names=[""],
    )
    generator.add_param(
        key="algo.actor_layer_dims",
        name="",
        group=-1,
        values=[[1024, 1024]],
        value_names=[""],
    )
    generator.add_param(
        key="algo.rnn.enabled",
        name="rnn",
        group=-1,
        values=[False],
        value_names=["F"],
    )


def set_no_gmm(generator, args):
    if not args.no_gmm:
        return 
    generator.add_param(
        key="algo.gmm.enabled",
        name="gmm",
        group=-1,
        values=[False],
        value_names=["F"],
    )

def set_parallel_rollout_envs(generator, args):
    generator.add_param(
        key='experiment.rollout.parallel_envs',
        name='',
        group=-10,
        values=[args.parallel_rollout_envs]
    )

def add_loss_keys(generator, args):
    if not args.use_loss_keys:
        return

    generator.add_param(
        key="train.dataset_keys",
        name="",
        group=3,
        values=[
            (
                "actions",
                "rewards",
                "dones",
                "action_modes",
                "intv_labels",
                "round",
            )
        ],
    )

def add_rollout_delete_ratio(generator, args):

    if args.delete_rollout_ratio < 0:
        return

    generator.add_param(
        key="algo.hc_weights.delete_rollout_ratio",
        name="del_rollout",
        group=333399,
        values=[
            args.delete_rollout_ratio,
        ],
    )

def set_num_eps(generator, args):
  
    if args.num_eps < 0:
        return

    if args.num_eps == 10000:
        # hack, for benchmarking
        generator.add_param(
            key="train.num_eps",
            name="eps",
            group=-1000000,
            values=[200, 400, 800, 1200], #[args.num_eps],
        )
    else:
        generator.add_param(
            key="train.num_eps",
            name="eps",
            group=-1,
            values=[args.num_eps],
        )

""" Dreamer """
def set_kl_weight(generator, args):
    generator.add_param(
        key="algo.wm.kl_weight",
        name="klw",
        group=-1,
        values=[args.kl_weight],
    )

def set_kl_balance(generator, args):
    generator.add_param(
        key="algo.wm.kl_balance",
        name="klb",
        group=-1,
        values=[args.kl_balance],
    )

def set_seq_length(generator, args):

    if args.seq_length < 0:
        return

    assert args.seq_length > 0

    generator.add_param(
        key="train.seq_length",
        name="seql",
        group=-1,
        values=[args.seq_length],
    )

def set_save(generator, args):

    if args.save < 0:
        return

    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="save",
        group=-1,
        values=[args.save],
    )

def set_reconst_image(generator, args):

    value_names = "recons"
    if "all" in args.recons_image:
        value_names += "_all"

        generator.add_param(
            key="algo.wm.output_image",
            name="",
            group=-1,
            values=[[]],
            value_names=[value_names],
        )
        return

    if "agentview_image" in args.recons_image:
        value_names += "_agent"
    if "robot0_eye_in_hand_image" in args.recons_image:
        value_names += "_eih"

    print("reconstructed image: ", value_names)

    generator.add_param(
        key="algo.wm.output_image",
        name="",
        group=-1,
        values=[args.recons_image],
        value_names=[value_names],
    )

def set_batch_size(generator, args):
    if args.bs < 0:
        return 

    generator.add_param(
        key="train.batch_size",
        name="bs",
        group=-1,
        values=[args.bs],
        value_names=[args.bs],
    )

def set_rgb_input(generator, args):

    value_names = "input"
    if "agentview_image" in args.input_image:
        value_names += "_agent"
    if "robot0_eye_in_hand_image" in args.input_image:
        value_names += "_eih"

    print("input image: ", value_names)

    generator.add_param(
        key="observation.modalities.obs.rgb",
        name="",
        group=-1,
        values=[args.input_image],
        value_names=[value_names],
    )

def set_gru_type(generator, args):

    generator.add_param(
        key="algo.wm.gru_type",
        name="",
        group=-1,
        values=[args.gru_type],
        value_names=[args.gru_type],
    )

def set_deter_dim(generator, args):

    generator.add_param(
        key="algo.wm.deter_dim",
        name="ddim",
        group=-1,
        values=[args.deter_dim],
    )

def set_hidden_dim(generator, args):

    generator.add_param(
        key="algo.wm.hidden_dim",
        name="",
        group=-1,
        values=[args.hidden_dim],
    )

def set_stoch_dim(generator, args):

    generator.add_param(
        key="algo.wm.stoch_dim",
        name="sdim",
        group=-1,
        values=[args.stoch_dim],
    )

def set_obs_embedding_dim(generator, args):

    generator.add_param(
        key="algo.wm.obs_embedding_dim",
        name="",
        group=-1,
        values=[args.obs_embedding_dim],
    )


def set_policy_weight(generator, args):

    generator.add_param(
        key="algo.wm.policy_weight",
        name="",
        group=-1,
        values=[args.policy_weight],
    )

def set_reward_weight(generator, args):

    generator.add_param(
        key="algo.wm.reward_weight",
        name="",
        group=-1,
        values=[args.reward_weight],
    )

def set_recons_weight(generator, args):
    generator.add_param(
        key="algo.wm.recons_weight",
        name="recons_w",
        group=-1,
        values=[args.recons_weight],
    )

def set_recons_zero_weight(generator, args):
    generator.add_param(
        key="algo.wm.recons_weight_zero",
        name="",
        group=-1,
        values=[args.recons_weight_zero],
    )

def set_prioritize_first_weight(generator, args):
    if args.prioritize_first_weight <= 1.0:
        return
        
    generator.add_param(
        key="algo.wm.prioritize_first_sampler",
        name="",
        group=-1,
        values=[True],
    ) 
    
    generator.add_param(
        key="algo.wm.prioritize_first_weight",
        name="pr_w",
        group=-1,
        values=[args.prioritize_first_weight],
    )

def set_image_output_activation(generator, args):
    if args.img_activation == "":
        return
    generator.add_param(
        key="algo.wm.image_output_activation",
        name="",
        group=-1,
        values=[args.img_activation],
    )

def set_initial(generator, args):
    generator.add_param(
        key="algo.wm.initial",
        name="",
        group=-1,
        values=[args.initial],
    )

def set_diff_weight(generator, args):
    generator.add_param(
        key="algo.wm.diff_weight",
        name="diff_w",
        group=-1,
        values=[args.diff_weight],
    )

def set_wm_class(generator, args):
    if args.wm_class == "":
        return
    
    assert args.wm_class in ["WorldModel", "WorldModelTwoAE"]
    name = "wm"
    if args.wm_class == "WorldModelTwoAE":
        name = "wm2ae"

    generator.add_param(
        key="algo.wm.wm_class",
        name="",
        group=-1,
        values=[args.wm_class],
        value_names=[name],
    )

def set_prior_larger(generator, args):
    if not args.prior_larger:
        return

    generator.add_param(
        key="algo.wm.prior_larger",
        name="",
        group=-1,
        values=[args.prior_larger],
    )

def set_free_bit(generator, args):

    generator.add_param(
        key="algo.wm.free_bit",
        name="free_bit",
        group=-1,
        values=[args.free_bit],
    ) 

def set_use_network_za(generator, args):

    generator.add_param(
        key="algo.wm.use_network_za",
        name="za",
        group=-1,
        values=[True if args.use_network_za else False],
    )

def set_obs_ld_dim(generator, args):
    if args.obs_ld_dim == 0:
        return

    generator.add_param(
        key="algo.wm.obs_ld_dim",
        name="obs_ld",
        group=-1,
        values=[args.obs_ld_dim],
    )

def set_stoch_only(generator, args):
    if not args.stoch_only:
        return

    generator.add_param(
        key="algo.wm.stoch_only",
        name="sonly",
        group=-1,
        values=[True],
    )

def set_diff_za_dim(generator, args):
    if not args.diff_za_dim:
        return

    generator.add_param(
        key="algo.wm.diff_za_dim",
        name="diff_za",
        group=-1,
        values=[True],
        value_names=["T"],
    )

def set_use_ld_decoder(generator, args):
    if not args.use_ld_decoder:
        return

    generator.add_param(
        key="algo.wm.use_ld_decoder",
        name="ld_decode",
        group=-1,
        values=[True],
        value_names=["T"],
    )

def set_pi_update_wm(generator, args):
    if not args.pi_update_wm:
        return

    generator.add_param(
        key="algo.wm.pi_update_wm",
        name="pi_update_wm",
        group=-1,
        values=[True],
        value_names=["T"],
    )

def set_bc_dreamer_configs(generator, args):

    """Set configs for BC Dreamer"""
    if args.use_reward:
        assert args.trainer_type == "combined"

    if args.trainer_type == "combined":
        assert not args.dyn_detach

    generator.add_param(
        key="algo.dyn.dyn_detach",
        name="",
        group=-1,
        values=[args.dyn_detach],
    )

    generator.add_param(
        key="algo.dyn.dyn_weight",
        name="dyn_w",
        group=-1,
        values=[args.dyn_weight],
    )

    generator.add_param(
        key="algo.dyn.combine_enabled",
        name="combined",
        group=-1,
        values=[True if args.trainer_type == "combined" else False],
        value_names=["T" if args.trainer_type == "combined" else "F"],
    )

    generator.add_param(
        key="algo.dyn.use_res_mlp",
        name="",
        group=-1,
        values=[args.use_res_mlp],
    )

    generator.add_param(
        key="algo.dyn.smooth_weight",
        name="smoo_w",
        group=-1,
        values=[args.smooth_weight],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.smooth_dynamics",
        name="use_smoo",
        group=-1,
        values=[args.smooth_dynamics],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.stochastic_inputs",
        name="",
        group=-1,
        values=[args.stochastic_inputs],
    )

    generator.add_param(
        key="algo.dyn.kl_balance",
        name="klb",
        group=-1,
        values=[args.kl_balance],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.dyn_class",
        name="d_class",
        hidename=True,
        group=-1,
        #values=[args.dyn_class],
        values=["vae"],
    )

    generator.add_param(
        key="algo.vae.prior.is_conditioned",
        name="prior_cond",
        hidename=True,
        group=-1,
        #values=[args.prior_is_conditioned],
        #value_names=["T" if args.prior_is_conditioned else "F"],
        values=[True],
        value_names=["T"],
    )

    generator.add_param(
        key="algo.vae.prior.use_gmm",
        name="prior_gmm",
        hidename=True,
        group=-1,
        values=[True if args.prior_use_gmm else False],
        value_names=["T" if args.prior_use_gmm else "F"],
    )


    if args.prior_gmm_dim > 0:
        generator.add_param(
            key="algo.vae.prior.gmm_num_modes",
            name="gmm_dim",
            group=-1,
            values=[args.prior_gmm_dim],
        )

    assert args.prior_use_categorical + args.prior_use_gmm <= 1

    generator.add_param(
        key="algo.vae.prior.use_categorical",
        name="prior_cat",
        group=-1,
        hidename=True,
        values=[True if args.prior_use_categorical else False],
        value_names=["T" if args.prior_use_categorical else "F"],
    )

    if args.prior_cat_dim > 0:
        generator.add_param(
            key="algo.vae.prior.categorical_dim",
            name="cat_dim",
            group=-1,
            values=[args.prior_cat_dim],
        )

    generator.add_param(
        key="algo.vae.enc_use_res_mlp",
        name="enc_resmlp",
        group=-1,
        #hidename=True,
        hidename=True,
        #values=[args.enc_use_res_mlp],
        #value_names=["T" if args.enc_use_res_mlp else "F"],
        values=[True],
        value_names=["T"],
    )

    generator.add_param(
        key="algo.vae.dec_use_res_mlp",
        name="dec_resmlp",
        group=-1,
        hidename=True,
        #values=[args.dec_use_res_mlp],
        #value_names=["T" if args.dec_use_res_mlp else "F"],
        values=[True],
        value_names=["T"],
    )

    if args.vae_latent_dim > 0:
        generator.add_param(
            key="algo.vae.latent_dim",
            name="vae_ldim",
            group=-1,
            values=[args.vae_latent_dim],
        )

    generator.add_param(
        key="algo.vae.kl_weight",
        name="vae_klw",
        group=-1,
        values=[args.vae_kl_weight],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.use_history",
        name="use_hist",
        group=-1,
        values=[True],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.use_real",
        name="use_real",
        group=-1,
        values=[False],
        hidename=True,
    )

    if args.sort_key:
        generator.add_param(
            key="train.sort_demo_key",
            name="key",
            group=-1,
            values=[
                "MFI"
                ],
        )

    if args.lr_dyn > 0:
        generator.add_param(
            key="algo.optim_params.dynamics.learning_rate.initial",
            name="lr_dyn",
            group=-1,
            values=[args.lr_dyn],
        )

    set_no_validate(generator, args)

    set_bc_dreamer_rewards(args, generator)

def set_bc_dreamer_rewards(args, generator):
    if not args.use_reward:
        return

    generator.add_param(
        key="algo.dyn.use_reward",
        name="use_rew",
        group=-1,
        values=[args.use_reward],
        value_names=["T" if args.use_reward else "F"],
        hidename=True,
    )

    generator.add_param(
        key="train.dataset_keys",
        name="",
        group=3,
        values=[
            (
            "actions",
            "action_modes",
            "sparse_reward",
            "dense_reward",
            "three_class",
            "intv_labels"
            )
        ],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.reward_weight",
        name="rew_weigh",
        group=-1,
        values=[args.reward_weight],
    )

    generator.add_param(
        key="algo.dyn.rew.rew_class",
        name="rew_bi",
        group=-1,
        values=[args.rew_class],
        value_names=[args.rew_class],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.rew.hidden_dim",
        name="rdim",
        group=-1,
        values=[args.rew_hidden_dim],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.rew.num_layers",
        name="rn",
        group=-1,
        values=[args.rew_num_layers],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.rew.activation",
        name="ract",
        group=-1,
        values=[args.rew_activation],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.rew.use_action",
        name="r_use_a",
        group=-1,
        values=[args.rew_use_action],
        value_names=["T" if args.rew_use_action else "F"]
    )

    generator.add_param(
        key="algo.dyn.rew.use_res_mlp",
        name="rresmlp",
        group=-1,
        values=[args.rew_use_res_mlp],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.rew.use_weighted_loss",
        name="use_w_loss",
        group=-1,
        values=[args.use_weighted_loss],
    )

    generator.add_param(
        key="algo.dyn.rew.lstm.enabled",
        name="rew_rnn",
        group=-1,
        #values=[args.rew_rnn],
        #value_names=["T" if args.rew_rnn else "F"]
        values=[True],
        value_names=["T"]
    )

    generator.add_param(
        key="algo.dyn.rew.lstm.dropout",
        name="rew_drop",
        group=-1,
        values=[args.rew_dropout],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.rew.lstm.bidirectional",
        name="rew_bidir",
        group=-1,
        values=[True if args.rew_bidirectional else False],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.rew.lstm.seq_length",
        name="rew_seq",
        group=-1,
        values=[args.rew_seq_length],
    )

    generator.add_param(
        key="algo.dyn.rew.all_seq_prediction",
        name="rew_all_pred",
        group=-1,
        values=[args.all_seq_prediction],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.use_policy",
        name="use_pol",
        group=-1,
        values=[not args.disable_policy],
        value_names=["F" if args.disable_policy else "T"],
        hidename=True,
    )

    generator.add_param(
        key="algo.dyn.use_dynamics",
        name="use_dyn",
        group=-1,
        values=[not args.disable_dynamics],
        value_names=["F" if args.disable_dynamics else "T"],
        hidename=True,
    )

    if args.binary_loss == "focal":
        generator.add_param(
            key="algo.dyn.rew.binary_loss",
            name="loss",
            group=-1,
            values=[args.binary_loss],
        )
    
        generator.add_param(
            key="algo.dyn.rew.focal_alpha",
            name="focal_alpha",
            group=-1,
            values=[args.focal_alpha],
        )

    if args.load_prev_policy_dyn:
        assert args.use_reward
        generator.add_param(
            key="algo.dyn.load_prev_policy_dyn",
            name="load",
            group=-1,
            values=[args.load_prev_policy_dyn],
            value_names=["T"]
        )

        generator.add_param(
            key="algo.dyn.load_ckpt",
            name="ckpt",
            group=-1,
            values=[args.load_ckpt],
            value_names=[args.load_ckpt_name],
        )
    
    if args.obs_sg:
        generator.add_param(
            key="algo.dyn.obs_sg",
            name="obs_sg",
            group=-1,
            values=[args.obs_sg],
            value_names=["T"]
        )

    if args.rew_fc_num > 0:
        generator.add_param(
            key="algo.dyn.rew.fc_num",
            name="fc_num",
            group=-1,
            values=[args.rew_fc_num],
        )


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        choices=['square', 'tool_hang',
                 'coffee_v0', 'threading_v0',
                 'gear_real', 'kcup_real',
                 'cleanup_real',
                 ],
        required=True,
    )

    # rollout parallel environments
    parser.add_argument(
        '--parallel_rollout_envs',
        type=int,
        default=1
    )

    parser.add_argument(
        "--use_joint",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--wandb_proj_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None
    )

    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None
    )

    parser.add_argument(
        '--iwr',
        action='store_true'
    )

    parser.add_argument(
        '--bootstrap',
        action='store_true'
    )

    parser.add_argument(
        '--ft',
        action='store_true'
    )

    parser.add_argument(
        '--debug',
        action='store_true'
    )

    parser.add_argument(
        '--tmplog',
        action='store_true'
    )

    parser.add_argument(
        '--modality',
        type=str,
        choices=['ld', 'im'],
        required=True
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )

    parser.add_argument(
        "--nr",
        type=int,
        default=-1
    )  # number of rollouts

    parser.add_argument(
        "--ns",
        type=int,
        default=-1
    )  # number of seeds

    parser.add_argument(
        "--script",
        type=str,
        default=None
    )

    parser.add_argument(
        "--large_scale",
        action="store_true",
    )

    parser.add_argument(
        "--mid_scale",
        action="store_true",
    )

    parser.add_argument(
        "--num_epoches",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--sweep_resmlp",
        action="store_true",
    )

    parser.add_argument(
        "--no_resmlp",
        action="store_true",
    )

    parser.add_argument(
        "--img_size128",
        action="store_true"
    )
    parser.add_argument(
        "--img_size240",
        action="store_true"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--sweep_iwr",
        action="store_true"
    )
    parser.add_argument(
        "--warmstart",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--gradnorm",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--no_eval",
        action="store_true",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=-1,
    )

    parser.add_argument(
        "--lr_dyn",
        type=float,
        default=-1,
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=-1,
    )

    parser.add_argument(
        "--full_pass",
        action="store_true"
    )

    parser.add_argument(
        "--not_use_sampler",
        action="store_true"
    )

    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--use_human_relabeled",
        action="store_true"
    )

    parser.add_argument(
        "--rollout_horizon",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--rollout_rate",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--reweight_rounds",
        action="store_true",
    )

    parser.add_argument(
        "--no_batch_norm",
        action="store_true",
    )

    parser.add_argument(
        "--round_resample",
        action="store_true",
    )

    parser.add_argument(
        '--rnn',
        action='store_true'
    )

    parser.add_argument(
        '--no-gmm',
        action='store_true',
    )

    parser.add_argument(
        '--no-rnn',
        action='store_true',
    )

    parser.add_argument(
        "--use_loss_keys",
        action="store_true",
    )

    parser.add_argument(
        "--num_eps",
        default=-1,
        type=int,
    )

    parser.add_argument(
        "--img_type",
        type=str,
        default=None,
        choices=[
            None, 
            "birdview",
            "birdview_vae",
            "113_84", 
            "240", 
            "180_84"
            ]
    )

    parser.add_argument(
        "--use_weight_sampler",
        action="store_true",
    )
    parser.add_argument(
        "--delete_rollout_ratio",
        type=float,
        default=-1.,
    )

    parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--kl_balance",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--seq_length",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--recons_image",
        type=str,
        nargs="+",
        default=["all"]
                 #"robot0_eye_in_hand_image"]
    )

    parser.add_argument(
        "--bs",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--input_image",
        type=str,
        nargs="+",
        default=["agentview_image",
                 "robot0_eye_in_hand_image"]
    )

    parser.add_argument(
        "--gru_type",
        type=str,
        default="gru",
        choices=["gru", "gru_layernorm"]
    )

    parser.add_argument(
        "--deter_dim",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--stoch_dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--obs_embedding_dim",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--policy_weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--recons_weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--recons_weight_zero",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--prioritize_first_weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--img_activation",
        type=str,
        default="",
    )

    parser.add_argument(
        "--initial",
        type=str,
        default="zeros",
    )

    parser.add_argument(
        "--diff_weight",
        action="store_true",
    )

    parser.add_argument(
        "--wm_class",
        type=str,
        default="",
    )

    parser.add_argument(
        "--prior_larger",
        action="store_true",
    )

    parser.add_argument(
        "--free_bit",
        default=0,
        type=float,
    )

    parser.add_argument(
        "--use_network_za",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--diff_za_dim",
        action="store_true",
    )

    parser.add_argument(
        "--stoch_only",
        action="store_true",
    )

    parser.add_argument(
        "--obs_ld_dim",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--use_ld_decoder",
        action="store_true",
    )

    parser.add_argument(
        "--pi_update_wm",
        action="store_true",
    )

    parser.add_argument(
        "--dyn_weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dyn_detach",
        action="store_true",
    )

    parser.add_argument(
        "--trainer_type",
        type=str,
        choices=["combined", "separate"],
        default="combined",
    )

    parser.add_argument(
        "--use_res_mlp",
        action="store_true",
    )
        
    parser.add_argument(
        "--smooth_weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--smooth_dynamics",
        action="store_true",
    )
   
    parser.add_argument(
        "--stochastic_inputs",
        action="store_true",
    )

    parser.add_argument(
        "--dyn_class",
        type=str,
        default="deter",
    )

    parser.add_argument(
        "--prior_is_conditioned",
        action="store_true",
    )

    parser.add_argument(
        "--prior_use_gmm",
        type=int,
        default=1,
    )
  
    parser.add_argument(
        "--prior_use_categorical",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--prior_cat_dim",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--prior_gmm_dim",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--enc_use_res_mlp",
        action="store_true",
    )

    parser.add_argument(
        "--dec_use_res_mlp",
        action="store_true",
    )

    parser.add_argument(
        "--vae_latent_dim",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--vae_kl_weight",
        type=float,
        default=0.00001,
    )

    # reward
    parser.add_argument(
        "--reward_weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--rew_hidden_dim",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--rew_num_layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--rew_activation",
        type=str,
        default="leaky_relu",
    )

    parser.add_argument(
        "--rew_class",
        type=str,
        default="binary",
        choices=["binary", "three_class", "dense"]
    )

    parser.add_argument(
        "--rew_use_action",
        action="store_true",
    )

    parser.add_argument(
        "--rew_use_res_mlp",
        action="store_true",
    )
   
    parser.add_argument(
        "--rew_rnn",
        action="store_true",
    )

    parser.add_argument(
        "--rew_dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--rew_bidirectional",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--rew_seq_length",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--all_seq_prediction",
        action="store_true",
    )
    
    parser.add_argument(
        "--use_reward",
        action="store_true",
    )

    parser.add_argument(
        "--disable_policy",
        action="store_true",
    )

    parser.add_argument(
        "--disable_dynamics",
        action="store_true",
    )

    parser.add_argument(
        "--use_weighted_loss",
        action="store_true",
    )

    parser.add_argument(
        "--sort_key",
        action="store_true",
    )

    parser.add_argument(
        "--save",
        default=-1,
        type=int,
    )


    parser.add_argument(
        "--binary_loss",
        default="bce",
        type=str,
    )

    parser.add_argument(
        "--focal_alpha",
        default=0.25,
        type=float,
    )

    # loading previous ckpt
    parser.add_argument(
        "--load_prev_policy_dyn",
        action="store_true",
    )

    parser.add_argument(
        "--load_ckpt",
        default="",
        type=str,
    )
   
    parser.add_argument(
        "--load_ckpt_name",
        default="",
        type=str,
    )

    parser.add_argument(
        "--obs_sg",
        action="store_true",
    )

    parser.add_argument(
        "--rew_fc_num",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--use_history",
        action="store_true",
    )

    parser.add_argument(
        "--use_real",
        action="store_true",
    )   

    parser.add_argument(
        "--no_val",
        action="store_true",
    )

    parser.add_argument(
        "--no_intv",
        action="store_true",
    )

    return parser

def make_generator(args, make_generator_helper, algo, dreamer=False, bc_dreamer=False):
    assert algo in ["awac", "bc", "iql"]

    if args.tmplog or args.debug:
        args.name = "debug"
    else:
        time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-')
        args.name = time_str + args.name

    # make config generator
    generator = make_generator_helper(args)

    assert not (args.iwr and args.sweep_iwr)

    """ Standard ones for all runs """
    add_rnn(generator, args)
    add_dataset(generator, args)
    set_no_gmm(generator, args)
    set_no_rnn(generator, args)
    set_num_rollouts(generator, args)
    set_num_seeds(generator, args)
    set_iwr_and_ft_settings(generator, args)
    set_dataset(generator, args)
    set_debug_mode(generator, args)
    set_exp_id(generator, args)
    set_env_settings(generator, args)
    set_wandb_mode(generator, args)
    set_resmlp(generator, args, algo=algo)
    no_resmlp(generator, args, algo=algo)
    set_img_size_128(generator, args)
    set_img_size_240(generator, args)
    set_iwr(generator, args)
    set_bootstrap(generator, args)
    set_sweep_iwr(generator, args)
    set_warmstart(generator, args)
    set_gradnorm(generator, args, algo=algo)
    set_no_eval(generator, args)
    set_large_scale_eval_mode(generator, args)
    set_mid_scale_eval_mode(generator, args)
    set_num_epochs(generator, args)
    set_learning_rate(generator, args)
    set_weight_decay(generator, args)
    set_full_pass(generator, args)
    set_use_sampler(generator, args)
    set_filter_key(generator, args)
    use_human_relabeled_weights(generator, args)
    set_horizon(generator, args)
    set_rollout_rate(generator, args)
    set_reweight_rounds(generator, args)
    set_round_resampling(generator, args)
    set_parallel_rollout_envs(generator, args)
    add_loss_keys(generator, args)
    use_weight_sampler(generator, args)
    add_rollout_delete_ratio(generator, args)
    set_num_eps(generator, args)
    set_batch_size(generator, args)
    set_seq_length(generator, args)
    set_save(generator, args)
    use_joint(generator, args)
    set_no_intv(generator, args)

    # dreamer
    if dreamer:
        set_kl_weight(generator, args)
        set_kl_balance(generator, args)
        set_seq_length(generator, args)
        set_reconst_image(generator, args)
        set_batch_size(generator, args)
        set_rgb_input(generator, args)
        set_gru_type(generator, args)
        set_deter_dim(generator, args)
        set_stoch_dim(generator, args)
        set_hidden_dim(generator, args)
        set_obs_embedding_dim(generator, args)
        set_policy_weight(generator, args)
        set_reward_weight(generator, args)
        set_recons_weight(generator, args)
        set_recons_zero_weight(generator, args)
        set_prioritize_first_weight(generator, args)
        set_image_output_activation(generator, args)
        set_initial(generator, args)
        set_diff_weight(generator, args)
        set_wm_class(generator, args)
        set_prior_larger(generator, args)
        set_free_bit(generator, args)
        set_use_network_za(generator, args)
        set_stoch_only(generator, args)
        set_diff_za_dim(generator, args)
        set_obs_ld_dim(generator, args)
        set_use_ld_decoder(generator, args)
        set_pi_update_wm(generator, args)

    elif bc_dreamer:
        set_bc_dreamer_configs(generator, args)

    # generate jsons and script
    generator.generate()
