import argparse
import subprocess
import time
import nvidia_smi
import os

def parse_command_lst(hp_script):
    """
    Helper script to parse the executable hyperparameter script generated from hyperparameter_helper.py (in batchRL)
    to infer the filepaths to the generated configs.
    Args:
        hp_script (str): Absolute fpath to the generated hyperparameter script
    Returns:
        list: Absolute paths to the configs to be deployed in the hp sweep
    """
    # Create list to fill as we parse the script
    commands = []
    # Open and parse file line by line
    with open(hp_script) as f:
        for line in f:
            # Make sure we only parse the lines where we have a valid python command
            if line.startswith("python"):
                # Extract only the config path
                commands.append(line)
    print(commands)
    # Return configs
    return commands

def execute_commands(commands_lst, memory_per_job, gpu_start):
    subprocesses = []
    gpu_choice = []
    
    for command in commands_lst:
        GPU_choice = find_next_gpu(memory_per_job, gpu_start)
        gpu_start = GPU_choice
        gpu_choice.append(gpu_start)
        
        command = command.split()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_choice)
        print("'CUDA_VISIBLE_DEVICES' is set to '{}'".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        with open("no.txt") as f:
            subprocesses.append(subprocess.Popen(command, stdin=f))
        time.sleep(1)

def find_next_gpu(memory_per_job, gpu_start):
    nvidia_smi.nvmlInit()
    
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(gpu_start, gpu_start + deviceCount):
        device_num = i % deviceCount
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_num)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        if info.free > memory_per_job:
            return device_num
    print("no more GPU available")
    exit()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--hp_script',
        required=True,
        type=str
    )

    parser.add_argument(
        '--memory_per_job',
        default=4000,
        type=int
    )

    parser.add_argument(
        '--gpu_start',
        default=1,
        type=int
    )

    args = parser.parse_args()
    commands = parse_command_lst(args.hp_script)
    execute_commands(commands, args.memory_per_job, args.gpu_start)
