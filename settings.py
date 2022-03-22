import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import math
import numpy as np
import random


# reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# hyper-parameters
MDP_settings = {
    "STATE_SIZE": 32 + 3,
    "ACTION_SIZE": 1 + 1,
    "GAMMA": 0.98,
    "EPSILON_START": 1,
    "EPSILON_DECAY": 0.5,
    "EPSILON_END": 0.01,
    "BATCH_SIZE": 32,
    "NUM_EPISODES": 10000,
}


agent_settings = {
    "EYE_SIZE": 3,
    "INPUT_SIZE": MDP_settings['STATE_SIZE'],
    "HID1_SIZE": 128,
    "HID2_SIZE": 64,
    "OUTPUT_SIZE": MDP_settings['ACTION_SIZE'],
    "LEARNING_RATE": 1e-3,
    "EXPERIENCE_REPLAY_CAPACITY": 256,
    "TARGET_UPDATE": 200
}
