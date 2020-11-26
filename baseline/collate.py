import os, sys

import numpy as np
import matplotlib.pyplot as plt

# dirs = os.listdir('csvs/')
dirs = [2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
rewards, stds = {}, {}
seeds = [0,2,4]
for dir in dirs:
    rm, rs = [], []
    for seed in seeds:
        
        with open('csvs/{}/{}/{}/{}/0.monitor.csv'.format(dir, sys.argv[1], sys.argv[2], seed),'r') as file:
            text = file.readlines()
            text = text[2:]
            vals = np.asarray([i.split(',')[0] for i in text]).astype(float)
            rm.extend(vals.tolist())
    mean_vals = np.mean(rm)
    rewards[dir] = mean_vals
    stds[dir] = np.std(rm)

# print(rewards)
fig, ax = plt.subplots()#figsize=(10,10))
ax.plot(list(rewards.keys()), np.array(list(rewards.values())))
ax.fill_between(stds.keys(), np.array(list(rewards.values())) + np.array(list(stds.values())), np.array(list(rewards.values())) - np.array(list(stds.values())), alpha=0.1, facecolor='b')
# ax.fill_between(stds.keys(), , alpha=0.5, facecolor='b')

ax.set_ylabel("Reward")
ax.set_xlabel("Bits")
ax.set_title("Sweet spot {} {}".format(sys.argv[1], sys.argv[2]))
plt.savefig("pngs/sweetspot_{}_{}.png".format(sys.argv[1], sys.argv[2]))
