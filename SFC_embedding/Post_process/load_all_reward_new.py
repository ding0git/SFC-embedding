import torch
from tasks import motsp
from tasks.motsp import TSPDataset, reward
from torch.utils.data import DataLoader
from model import DRL4TSP
from trainer_motsp_transfer import StateCritic
import numpy as np
import os
import matplotlib.pyplot as plt
from Post_process.dis_matrix import dis_matrix
import time

# Load the trained model and convert the obtained Pareto Front to the .mat file.
# It is convenient to visualize it in matlab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# "../tsp_transfer_100run_500000_5epoch_20city/20"效果一般。应该再训练一遍
save_dir = "../tsp_transfer/24"
# save_dir = "../tsp_transfer/100"
# param
update_fn = None
STATIC_SIZE = 32  # (x, y)
DYNAMIC_SIZE = 1  # dummy for compatibility

# claim model
actor = DRL4TSP(STATIC_SIZE,
                DYNAMIC_SIZE,
                256,
                update_fn,
                motsp.update_mask,
                1,
                0.1).to(device)
critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, 256).to(device)

# data 143

train_data = TSPDataset(24, 1, 12345)
train_loader = DataLoader(train_data, 1, False, num_workers=0)
ac = os.path.join(save_dir, "w_%2.2f_%2.2f" % (1.00, 0.00), "actor.pt")
cri = os.path.join(save_dir, "w_%2.2f_%2.2f" % (1.00, 0.00), "critic.pt")
actor.load_state_dict(torch.load(ac, device))
critic.load_state_dict(torch.load(cri, device))

def getchain(fname):
    chain = []
    with open(fname, 'r+', encoding='utf-8') as f:
        line = f.readline()
        num = list(map(int, line.split(',')))
        while line:
            num = list(map(int, line.split(',')))
            while len(num) == 1:
                line = f.readline()
                num = list(map(int, line.split(',')))

            p = num[0]
            n = num[1]
            chaintem = []
            for i in range(p + n):
                line = f.readline()
                num = list(map(int, line.split(',')))
                if i < int(p):
                    chaintem.append([int(num[1]), int(num[2]), 0])
                else:
                    chaintem[i - p][2] = int(num[3])
            for u in chaintem:
                chain.append(u)
            line = f.readline()

    return chain

for batch_idx, batch in enumerate(train_loader):
    print(batch_idx)
    static, dynamic, x0 = batch

    static = static.to(device)
    dynamic = dynamic.to(device)
    x0 = x0.to(device) if len(x0) > 0 else None

    # Full forward pass through the dataset
    chain_b = [getchain("../txt/45sfcs.txt")]
    chain_b=torch.tensor(chain_b)
    with torch.no_grad():
        tour_indices, tour_logp = actor(chain_b, static, dynamic, x0)
        reward, obj1, obj2 = reward(chain_b, tour_indices, 1, 0)
        print(reward/100)



