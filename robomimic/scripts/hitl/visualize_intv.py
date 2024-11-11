import os
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats
import seaborn as sns

sns.set_theme(style="darkgrid")

# Square
square_data = {
"ours_data": {
    "policy": [0.333, 0.782, 0.931, 0.958],
    "intv": [0, 2306, 2306 * 2, 2306 * 3],
    "intv_ratio": [0.126133, 0.0289273, 0.0135575],
    "round": ["0", "1", "2", "3"],
    "error_bar": [0.055, 0.023, 0.007, 0.0103],

},

"iwr_data": {
    "policy": [0.333, 0.614, 0.8, 0.89],
    "intv": [0, 2306, 2306 * 2, 2306 * 3],
    "intv_ratio": [0.126133, 0.0594514, 0.0277264],
    "round": ["0", "1", "2", "3"],
    "error_bar": [0.055, 0.043, 0.018, 0.026],
},
"bc_data": {
    "policy": [0.333, 0.6266666666666666, 0.8733333333333334, 0.9166666666666666],
    'error_bar': [0, 0.016996731711975962, 0.016996731711975962, 0.015634719199411447],
},
"iql_data": {
     "policy": [0.333, 0.26333333333333336, 0.49333333333333335, 0.723333],
     'error_bar': [0, 0.016996731711975965, 0.016996731711975965, 0.041],
},
    
"ablations": {
    "no_demos": {
        "policy": [0.7466666666666666, 0.9188888888888889, 0.9544444444444444, ],
        "error_bar": [0.045946829173634095, 0.021829869671542768, 0.013425606637327314, ],
    }
    ,
    "no_preintv": {
        "policy": [0.7266666666666666, 0.931, 0.9522222222222223, ],
        "error_bar": [0.02867441755680878, 0.008975274678557464, 0.011331154474650619, ],
    }
    ,
    "no_intv": {
        "policy": [0.6555555555555554, 0.9033333333333333, 0.9400000000000001],
        "error_bar": [0.035935470286213834, 0.026246692913372675, 0.008164965809277223,]
    }
}
}

colors_ablations = {
    "IWR": "dodgerblue",
    "Ours": "coral",
    "no_demos": "peru",
    "no_preintv": "steelblue",
    "no_intv": "limegreen",
}

def plot_ablations(task, title):
    colors = colors_ablations

    iwr_data = task["iwr_data"]
    ours_data = task["ours_data"]
    ablations = task["ablations"]

    fig, ax = plt.subplots()
    
    plt.plot(ours_data["round"][1:], 
             np.array(ours_data["policy"][1:]) * 100,
             marker='o', label="Ours", color=colors["Ours"],)
    
    plt.fill_between(ours_data["round"][1:], 
                     np.array(ours_data["policy"][1:]) * 100 - np.array(ours_data["error_bar"][1:]) * 100, 
                     np.array(ours_data["policy"][1:]) * 100 + np.array(ours_data["error_bar"][1:]) * 100, 
                     alpha=0.2,
                     color=colors["Ours"],
                     )

    
    plt.errorbar(ours_data["round"][1:], 
                np.array(ablations["no_demos"]["policy"]) * 100,
                yerr=np.array(ablations["no_demos"]["error_bar"]) * 100, marker="o", 
                label="No demo class", color=colors["no_demos"], linestyle='dashed',)
    
    plt.errorbar(ours_data["round"][1:], 
                np.array(ablations["no_preintv"]["policy"]) * 100,
                yerr=np.array(ablations["no_preintv"]["error_bar"]) * 100, marker="o", 
                label="No preintv class", color=colors["no_preintv"], linestyle='dashed',)
    
    plt.errorbar(ours_data["round"][1:], 
                np.array(ablations["no_intv"]["policy"]) * 100,
                yerr=np.array(ablations["no_intv"]["error_bar"]) * 100, marker="o", 
                label="No intv class", color=colors["no_intv"], linestyle='dashed',)
    
    ax.set_xlabel("Rounds", fontsize=16)
    ax.set_ylabel("Success Rate (%)", fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.set_ylim(60, 100)
    
    plt.xticks(size = 16)
    plt.yticks(size = 16)
    
    #plt.legend(labels=['Ours', 'IWR', "BC-RNN"])
    print(plt.rcParams['font.family'])
    plt.legend(loc="lower right", prop={'size': 16})
    fig.savefig("fig_{}.pdf".format(title), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
    )

    parser.add_argument(
        "--name",
        type=str,
    )

    colors = ["dodgerblue", "coral"]
    #colors = ["peachpuff", "sandybrown", "coral"]
    args = parser.parse_args()

    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (5.6, 4.8)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100

    labels = ["IWR (Round 3)", "Ours (Round 3)"]
    #labels = ["Ours (Round 1)", "Ours (Round 2)", "Ours (Round 3)"]

    for i in range(len(args.datasets)):
        dataset = args.datasets[i]
        out_dataset_path = os.path.expanduser(dataset)

        f = h5py.File(out_dataset_path, "r+")

        #demos = sorted(list(f["data"].keys()))
        demos = list(f["data"].keys())
        
        PAD_TOTAL = 1000

        import random
        random.shuffle(demos)

        count_set = np.array([])

        intv_sum_array = np.zeros((PAD_TOTAL,))

        total_samples = 0

        for ep in demos[:1000]:
            # store trajectory
            ep_data_grp = f["data/{}".format(ep)]
            action_modes = ep_data_grp["action_modes"][()]
            if (action_modes == -1).all():
                continue
            traj_length = len(action_modes)
            action_modes = np.pad(action_modes, (0, max(PAD_TOTAL - traj_length, 0)), 'constant')
            intv_sum_array += action_modes
            total_samples += traj_length

        internal = 5
        length = 500
        intv_sum_array /= total_samples
        intv_sum_array = intv_sum_array[:length]
        intv_sum_array = intv_sum_array.reshape((-1, internal)).sum(axis=1)
        plt.bar(x=range(int(length / internal)), height=intv_sum_array * 100, width=1, align='center', alpha=0.7,
                edgecolor="gray", label=labels[i]
                )

        #kde = stats.gaussian_kde(intv_sum_array)
        #plt.plot(kde(range(60)))
        print("Done.")
        f.close()

    plt.xticks(range(0, int(length / internal) + 1, 10),
               np.arange(0, length + 1, 50),
               size=16,
               )
    plt.yticks(
            np.arange(0, 30, 5) / 100,
            size = 16)

    ax.set_xlabel("Trajectory Timestep (t)", fontsize=16)
    ax.set_ylabel("Human Intervention Ratio (%)", fontsize=16)
    ax.set_title("Human Intervention Distribution", fontsize=20)

    #from matplotlib.ticker import FormatStrFormatter
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    #plt.legend(labels=['Ours', 'IWR', ])
    plt.legend(loc="upper right", prop={'size': 28})

    plt.legend()
    size = fig.get_size_inches()*fig.dpi
    print(fig.get_size_inches())
    print(fig.dpi)
    print(size)
    plt.savefig(args.name + ".pdf", bbox_inches='tight')

    plot_ablations(square_data, "Policy Performance")

