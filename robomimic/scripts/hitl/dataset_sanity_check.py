import h5py
import sys

dataset = sys.argv[1]

f = h5py.File(dataset, "r")
print(f["data"].keys())
for k in range(1, 200, 40):
    data = f["data/demo_{}".format(k)]
    rewards = data['rewards'][()]
    print("len: ", len(rewards))
    """
    print("rewards: ")
    print(rewards)
    dones = data['dones'][()]
    print()
    print("terminals: ")
    print(dones)
    obs = data['obs']
    print()
    print("obs keys: ")
    print(obs.keys())
    
    print(data.keys())
    """

success = 0
for k in range(len(f["data"].keys())):
    data = f["data/demo_{}".format(k)]
    rewards = data['rewards'][()]
    success += rewards[-1]

print("success rate: ", success / len(f["data"].keys()))
