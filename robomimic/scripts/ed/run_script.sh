DATASET_PATH="/home/shivin/hitl/experiments/square/rollouts/$1"
RENDER_IMAGE_NAMES="agentview"
N=100
DETECTOR_TYPE=$2
DETECTOR_CHECKPOINTS=None

DATASET_PATH_EXCEPT_LAST_5=${DATASET_PATH:0:-5}
VIDEO_PATH=${DATASET_PATH:0:-5}_$2_$N.mp4

DEMO_EMBEDDING="/home/shivin/hitl/embed_all_all.npy"
THRESHOLD=$3

ERROR_STATE_DIR="/home/shivin/hitl/experiments/square/rollouts/"$4

# Generate Python command
PYTHON_COMMAND="python ~/hitl/robomimic-hitl/robomimic/scripts/ed/playback_dataset_with_plots.py \
--dataset ${DATASET_PATH} --render_image_names ${RENDER_IMAGE_NAMES} --video_path ${VIDEO_PATH} \
--n ${N} --detector_type ${DETECTOR_TYPE} --detector_checkpoints ${DETECTOR_CHECKPOINTS} \
--demos_embedding_path ${DEMO_EMBEDDING} --threshold ${THRESHOLD} --error_state_dir ${ERROR_STATE_DIR}"

# Print the generated command
echo "Generated Python command:"
echo "${PYTHON_COMMAND}"
