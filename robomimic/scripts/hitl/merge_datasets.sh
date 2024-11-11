#!/bin/bash

## datasets to process
#DATASETS=(
#  "/home/soroushn/research/robomimic-hitl/datasets/square/0123_1728/round0.hdf5"
#  "/home/soroushn/research/robomimic-hitl/datasets/square/0123_1728/round1_0124_1803_50demos.hdf5"
#  "/home/soroushn/research/robomimic-hitl/datasets/square/0123_1728/round2.hdf5"
#  "/home/soroushn/research/robomimic-hitl/datasets/square/0123_1728/round3.hdf5"
#)
#
## where to store merged hdf5
#MERGED_DATASET_PATH="/home/soroushn/research/robomimic-hitl/datasets/square/0123_1728/round0_round1_0124_1803_50demos_round2_round3.hdf5"
#SKIP_MERGE=true
#
## experiment settings
#ENV="square"
#EXPNAME="bc_0123_1728_round0123"
#IWR=false
#FINETUNE=false
#CKPT_PATH="/home/soroushn/research/robomimic-hitl/expdata/hitl/square/image/bc_iwr_ft/bc_0123_1728_round012/2022-01-25-12-06-08/models/model_epoch_150.pth"
#DEBUG=true
#RNN=true

# where to store merged hdf5
MERGED_DATASET_PATH="/home/soroushn/research/robomimic-hitl/datasets/threading_v0/0126_2024/round0.hdf5"
SKIP_MERGE=true

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
