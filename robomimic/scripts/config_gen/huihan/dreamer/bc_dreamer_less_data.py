import os

import robomimic.utils.hyperparam_utils as HyperparamUtils
from robomimic.scripts.config_gen.helper import *
from robomimic.scripts.config_gen.huihan.jul.ours_percentage import *

def make_generator_helper(args):
    config_file = os.path.join(base_path, 'robomimic/exps/dreamer/bc_dreamer.json')

    exp_type = 'bc_dreamer'

    # set wandb project name
    if args.wandb_proj_name is None:
        strings = [exp_type, args.env, args.modality, args.name]
        args.wandb_proj_name = '_'.join([s for s in strings if s is not None])

    # initialize generator
    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file,
        generated_config_dir=os.path.join(os.getenv("HOME"), 'tmp', 'hitl_autogen_configs'),
        wandb_proj_name=args.wandb_proj_name,
        script_file=args.script,
    )

    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "../expdata/hitl/{env}/{modality}/{exp_type}".format(
                env=args.env,
                modality=args.modality,
                exp_type=exp_type,
            )
        ],
    )

    generator.add_param(
        key="train.num_eps",
        name="eps",
        group=-1,
        values=[
            500,
            ],
    )
   
    generator.add_param(
        key="train.sort_demo_key",
        name="key",
        group=-111,
        values=[
            "MFI"
            ],
    )

    use_category_ratio(generator, args)
    set_weight_params(generator, args)
    relabeling_type(generator, args)
    set_preintv_hardcode_relabel(generator, args)
    set_weighted_sampling(generator, args)

    return generator

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    make_generator(args, make_generator_helper, algo="bc", dreamer=False, bc_dreamer=True)
