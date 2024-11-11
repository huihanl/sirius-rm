import numpy as np
import h5py
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
#from collect_hitl_demos_ed import detector_from_config
import json
from datetime import datetime

from error_detectors import *


def get_obs_at_idx(obs, i):
    d = dict()
    for key in obs:
        d[key] = obs[key][i]
    return d


def load_detector_config(detector_type):
    try:
        config_path = os.path.join(
            os.path.dirname(__file__), "detector_configs/{}.json".format(detector_type)
        )
        with open(config_path) as f:
            detector_config = json.load(f)

        if 'CONFIG_OVERRIDE' in os.environ:
            config_override = json.loads(os.environ['CONFIG_OVERRIDE'])
            detector_config.update(config_override)
    except:
        detector_config = None

    return detector_config


def detector_from_config(detector_type, detector_checkpoints, policy_checkpoint=None):

    detector_config = load_detector_config(detector_type)

    if detector_type == "momart":
        detector = VAEMoMaRT(detector_checkpoints[0], 0.05)
    elif detector_type == "vae_goal":
        detector = VAEGoal(detector_checkpoints[0], 0.005)
    elif detector_type == "vae_action":
        detector = VAE_Action(detector_checkpoints[0], 0.05)
    elif detector_type == "ensemble":
        assert len(args.detector_checkpoints) >= 3
        detector = Ensemble(detector_checkpoints, 0.1)
    elif detector_type == "bc_dreamer_failure":
        detector = BCDreamer_Failure(detector_checkpoints[0], **detector_config)
    elif detector_type == "bc_dreamer_ood":
        detector = BCDreamer_OOD(detector_checkpoints[0], **detector_config)
    elif detector_type == "bc_dreamer_svm":
        detector = BCDreamer_SVM(detector_checkpoints[0], **detector_config)
    elif detector_type == "pato":
        detector = PATO(detector_checkpoints, vae_goal_th=0.005, ensemble_th=0.1)
    elif detector_type == "bc_dreamer_combined":
        detector = BCDreamer_Combined(detector_checkpoints, **detector_config)
    elif detector_type == "thrifty":
        detector = ThriftyDAggerED(detector_checkpoints, **detector_config)
    else:
        raise NotImplementedError

    return detector


def create_root_dir(args):
    # Get current date and time
    now = datetime.now()
    # Format as a string with hyphens
    now_str = now.strftime("%m-%d-%H-%M-%S")
        
    # Obtain checkpoint epoch
    file_name = os.path.basename(args.detector_checkpoints[0])
    parts = file_name.split("_")
    epoch_num = "_".join(parts[-2:]).split('.')[0]
        
    # Check if 'CONFIG_OVERRIDE' exists in the environment
    config_override = ""
    if 'CONFIG_OVERRIDE' in os.environ:
        config_override_json = json.loads(os.environ['CONFIG_OVERRIDE'])
        def dict_to_string(d):
            return "_".join(f"{key}_{value}" for key, value in d.items())
        config_override = dict_to_string(config_override_json)
        
    root_dir = f"{args.detector_type}_{epoch_num}_{config_override}_{now_str}"
    os.makedirs(root_dir, exist_ok=True)
    return root_dir

# Argparse setup
parser = argparse.ArgumentParser(description='Evaluate error detector')
parser.add_argument("--detector_type", type=str, 
                    choices=["momart", "ensemble", "vae_action", "vae_goal", 
                             "bc_dreamer_failure", "bc_dreamer_ood", "bc_dreamer_svm", "bc_dreamer_combined",
                             "dyn_ensemble", "pato", "thrifty"])
parser.add_argument("--detector_checkpoints", type=str, nargs='+')
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

root_dir = create_root_dir(args)

# Save as JSON
args_dict = vars(args)
with open('{}/configs.json'.format(root_dir), 'w') as f:
    json.dump(args_dict, f)

detector = detector_from_config(args.detector_type, args.detector_checkpoints)
detector.shadowing_node = True

# Load the dataset
f = h5py.File(args.dataset, "r")
demos = list(f["data"].keys())
inds = np.argsort([int(elem[5:]) for elem in demos])
demos = [demos[i] for i in inds]

predicted_intervention_points_all = []
true_intervention_points_all = []

error_results = {}

failure_trajs = {}

# Evaluate each demonstration
for ind in range(len(demos)):
    
    detector.reset()
    
    predicted_intervention_points = []
    true_intervention_points = []
    
    ep = demos[ind]
    obs = f["data/{}/obs".format(ep)]
    action_modes = f["data/{}/intv_labels".format(ep)][()]
    is_demo = (f["data/{}/action_modes".format(ep)][()] == -1).all()
    
    if is_demo:
        print("skip demo")
        continue
 
    obs_buffer = []
    for i in range(len(action_modes)):
        obs_i = get_obs_at_idx(obs, i)
        
        obs_buffer.append(obs_i)
        
        if detector.human_intervene(obs_buffer):
            predicted_intervention_points.append(1)
            predicted_intervention_points_all.append(1)
        else:
            predicted_intervention_points.append(0)
            predicted_intervention_points_all.append(0)

        if action_modes[i] == -10:  # replace 1 with the correct value for human intervention
            true_intervention_points.append(1)
            true_intervention_points_all.append(1)
        else:
            true_intervention_points.append(0)
            true_intervention_points_all.append(0)

    error_results[ep] = {
        "predicted": predicted_intervention_points,
        "true": true_intervention_points
    }
    
    if isinstance(detector, BCDreamer_Failure):
        failure_trajs[ep] = {
            "reward_label": detector._value_history["reward_label"],
            "reward_prob": detector._value_history["reward_prob"],
        }
        np.save("{}/failure_trajs".format(root_dir), failure_trajs)
        print(failure_trajs)
    elif isinstance(detector, BCDreamer_Combined):
        failure_trajs[ep] = {
            "reward_label": detector.rew_value_history["reward_label"],
            "reward_prob": detector.rew_value_history["reward_prob"],
        }
        np.save("{}/failure_trajs".format(root_dir), failure_trajs)
        print(failure_trajs)
    elif isinstance(detector, BCDreamer_OOD) or \
            isinstance(detector, BCDreamer_SVM) or \
            isinstance(detector, PATO) or \
            isinstance(detector, VAEMoMaRT) or \
            isinstance(detector, ThriftyDAggerED):
        
        failure_trajs[ep] = {
            "pred_value": detector._value_history
        }
        np.save("{}/failure_trajs".format(root_dir), failure_trajs)
        print(failure_trajs)


    np.save("{}/error_results".format(root_dir), error_results)
    print(error_results)

    # Calculate metrics
    cm = confusion_matrix(true_intervention_points, predicted_intervention_points)
    accuracy = accuracy_score(true_intervention_points, predicted_intervention_points)
    precision = precision_score(true_intervention_points, predicted_intervention_points)
    recall = recall_score(true_intervention_points, predicted_intervention_points)

    # Print metrics
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Visualize the results
    plt.figure(figsize=(20,10))
    plt.plot(true_intervention_points, label='True intervention', color='blue')
    plt.plot(predicted_intervention_points, label='Predicted intervention', color='red', linestyle='--')
    plt.xlabel('Time steps')
    plt.ylabel('Intervention needed')
    plt.title('True vs Predicted Intervention Points')
    plt.legend()
    plt.savefig('{}/{}.png'.format(root_dir, ep))
    #plt.show()

# Calculate metrics
cm = confusion_matrix(true_intervention_points_all, predicted_intervention_points_all)
accuracy = accuracy_score(true_intervention_points_all, predicted_intervention_points_all)
precision = precision_score(true_intervention_points_all, predicted_intervention_points_all)
recall = recall_score(true_intervention_points_all, predicted_intervention_points_all)

# Print metrics
print("++++")
print("Final metrics: ")
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
