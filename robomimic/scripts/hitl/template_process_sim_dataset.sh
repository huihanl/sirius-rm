#!/bin/bash

RAW_DATASET=$1

OUTPUT_DATASET=$2

IMG_SIZE=$3

CAMERA=$4

python ../conversion/convert_robosuite.py \
--dataset ${RAW_DATASET}

if [ $# -eq 2 ]

then

echo "==================================="
echo "======== Do not save image ========"
echo "==================================="

python ../dataset_states_to_obs.py \
--dataset ${RAW_DATASET} \
--output_dataset ${OUTPUT_DATASET} \

else

python ../dataset_states_to_obs.py \
--dataset ${RAW_DATASET} \
--output_dataset ${OUTPUT_DATASET} \
--camera_names ${CAMERA} robot0_eye_in_hand \
--camera_height ${IMG_SIZE} --camera_width ${IMG_SIZE} \
--am_value 0

fi
