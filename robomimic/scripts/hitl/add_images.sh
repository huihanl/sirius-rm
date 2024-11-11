#!/bin/bash

RAW_DATASET=$1.hdf5

OUTPUT_DATASET=$1_images.hdf5

ROBOMIMIC_DIR="/scratch/cluster/huihanl/robomimic-hitl"

python dataset_states_to_obs.py \
--dataset ${RAW_DATASET} \
--output_dataset ${OUTPUT_DATASET} \
--camera_names agentview robot0_eye_in_hand \
--camera_height 84 --camera_width 84
