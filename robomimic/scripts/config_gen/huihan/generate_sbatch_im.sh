#!/bin/bash

# experiment variables
ALGO="$1"
ENVS=(
  $4
)
MODALITY="im"
NAME="hitl-huihan"
WANDB_PROJ_NAME=$2
DEBUG=false
TMPLOG=false
NO_WANDB=false
ADDL_ARGS=$5

KICK_OFF_SBATCH=$3
N_EXPS_PER_INSTANCE=4
NUM_CPU=4
MEM_GB=16

PARTITION=titans
#PARTITION=dgx

ROBOMIMIC_DIR="/scratch/cluster/huihanl/robomimic-hitl"
#ROBOMIMIC_DIR="/home/soroushn/research/robomimic-hitl"

GEN_DIR_ROOT="/scratch/cluster/huihanl/tmp/autogen_configs/hitl"
#GEN_DIR_ROOT="/home/soroushn/tmp/autogen_configs/hitl"

MUJOCO_DIR="/u/huihanl/.mujoco/mujoco200_linux/bin"
EXECUTABLE_LOG_DIR="/scratch/cluster/huihanl/logs/hitl"                    # Where to log outputs / errors of sbatch script
if [ ${PARTITION} == "titans" ]
then
  EXCLUDE="titan-5" #"titan-5,titan-12"                                          # Specific machines to avoid using
else
  EXCLUDE=""                                          # Specific machines to avoid using
fi

# environment variables
PROJ_NAME="hitl"                                                      # Name of project
INTERPRETER="rpl"                                     # Which conda / venv to use for sbatch script

# sbatch variables (per script)
MAX_HRS=72                                                          # Max hours this script is allowed to run
EXTRA_PYTHONPATH=""                                     # Extra paths added to $PYTHONPATH when executing sbatch script
NOTIFICATION_EMAIL="huihanl@utexas.edu"                            # Email to send slurm notifications to (i.e.: when sbatch script finishes)
SHELL_SOURCE_SCRIPT="~/.bashrc"                      # Bash script to source at beginning of sbatch execution

# other settings
GPU_TYPE=any                                                        # Optionally specify a specific GPU type (titanx, titanrtx, etc...)
NUM_GPU=1

# YOU SHOULDN'T HAVE TO TOUCH ANYTHING BELOW HERE :)
###################################################

if [ ${DEBUG} == true ]
then
  ADDL_ARGS="${ADDL_ARGS} --debug"
fi

if [ ${TMPLOG} == true ]
then
  ADDL_ARGS="${ADDL_ARGS} --tmplog"
fi

if [ ${NO_WANDB} == true ]
then
  ADDL_ARGS="${ADDL_ARGS} --no_wandb"
fi

if ! [ -z "$WANDB_PROJ_NAME" ]
then
  ADDL_ARGS="${ADDL_ARGS} --wandb_proj_name ${WANDB_PROJ_NAME}"
fi


# Run the training script
conda activate ${INTERPRETER}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MUJOCO_DIR}


for env in "${ENVS[@]}"
do
    now="$(date +"%m-%d-%Y-%T")"

    # Compose path suffix
    gen_dir=${GEN_DIR_ROOT}/${ALGO}/${env}/${MODALITY}/${NAME}/${now}
    hp_sweep_script=${gen_dir}/hp_sweep.sh

    # 1. Run hyperparameter helper script
    python ${ROBOMIMIC_DIR}/robomimic/scripts/config_gen/huihan/${ALGO}.py \
    --env ${env} \
    --name ${NAME} \
    --modality ${MODALITY} \
    --script ${hp_sweep_script} \
    ${ADDL_ARGS}

    # 2. Generate sbatch scripts from hp_sweep script
    python ${ROBOMIMIC_DIR}/robomimic/scripts/slurm/run_hp_sweep.py \
    --hp_sweep_script ${hp_sweep_script} \
    --n_exps_per_instance ${N_EXPS_PER_INSTANCE} \
    --script ${ROBOMIMIC_DIR}/robomimic/scripts/train.py \
    --generated_dir ${gen_dir} \
    --python_interpreter ${INTERPRETER} \
    --partition ${PARTITION} \
    --exclude "${EXCLUDE}" \
    --gpu_type ${GPU_TYPE} \
    --num_gpu ${NUM_GPU} \
    --num_cpu ${NUM_CPU} \
    --mem_gb ${MEM_GB} \
    --max_hours ${MAX_HRS} \
    --executable_log_dir ${EXECUTABLE_LOG_DIR} \
    --shell_source_script ${SHELL_SOURCE_SCRIPT} \
    --notification_email "${NOTIFICATION_EMAIL}" \
    --mujoco_dir ${MUJOCO_DIR} \
    --overwrite False

    # 3. Optionally kick off all these runs
    if [ ${KICK_OFF_SBATCH} == true ]
    then
        for f in ${gen_dir}/*.sbatch
        do
            sbatch "${f}"
        done
    fi
done
