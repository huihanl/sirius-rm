#!/bin/bash

# datasets to process
DATASETS=(
  "/home/soroushn/research/robomimic-dev/datasets/hitl/push_real/0118_fork_small_ori.hdf5"
  "/home/soroushn/research/robomimic-dev/datasets/hitl/push_real/push_cake_round1_0121.hdf5"
)

# where to store merged hdf5
MERGED_DATASET_PATH="/home/soroushn/tmp/test.hdf5"
SKIP_MERGE=true

# experiment settings
ENV="push_real"
EXPNAME="test"
IWR=false
FINETUNE=false
CKPT_PATH="/home/soroushn/research/robomimic-hitl/expdata/hitl/push_real/image/bc/bc_push_real/2022-01-20-20-38-20/models/model_epoch_300.pth"
DEBUG=true

# misc settings
ROBOMIMIC_DIR="/home/soroushn/research/robomimic-hitl"

######## Don't modify anything beyond this point ########
if [ ${SKIP_MERGE} == false ]
then
#  # processing datasets
#  for dataset in "${DATASETS[@]}"
#  do
#    python ${ROBOMIMIC_DIR}/robomimic/scripts/hitl/process_real_robot_dataset.py \
#    --dataset ${dataset}
#
#    python ${ROBOMIMIC_DIR}/robomimic/scripts/hitl/add_action_modes.py \
#    --dataset ${dataset}
#  done

  # merging datasets
  python ${ROBOMIMIC_DIR}/robomimic/scripts/hitl/merge_datasets.py \
  --datasets ${DATASETS[@]} \
  --output_dataset ${MERGED_DATASET_PATH}
else
  echo "WARNING: Skip processing datasets!"
fi

# generate the config
ADDL_ARGS=""
if [ ${DEBUG} == true ]
then
  ADDL_ARGS="${ADDL_ARGS} --debug"
fi

if [ ${IWR} == true ]
then
  ADDL_ARGS="${ADDL_ARGS} --iwr"
fi

if [ ${FINETUNE} == true ]
then
  ADDL_ARGS="${ADDL_ARGS} --ft"
fi


python ${ROBOMIMIC_DIR}/robomimic/scripts/config_gen/bc_hitl.py \
--env ${ENV} \
--name ${EXPNAME} \
--dataset ${MERGED_DATASET_PATH} \
--ckpt_path ${CKPT_PATH} \
${ADDL_ARGS}
