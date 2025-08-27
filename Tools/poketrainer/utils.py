from torch.utils.data import Dataset, random_split, WeightedRandomSampler
from datetime import datetime
import matplotlib.pyplot as plt
import os
import logging
from torchvision import transforms
import torch
import platform
import re
import numpy as np

def log_view(log_path=LOG_ROOT_PATH):
    with open(log_path, 'r') as log_file:
        data_dict = {}
        for line in log_file:
            if data_dict == {}:
                pattern = r'(\w+): ([\d.]+)'
                matches = re.findall(pattern, line)
                data_dict = {key: [] for key, value in matches}
            pattern = r'(\w+): ([\d.]+)'
            matches = re.findall(pattern, line)
            for key, value in matches:
                if key == 'Epoch':
                    data_dict[key].append(int(value))
                else:
                    data_dict[key].append(float(value))

    epoch = data_dict['Epoch']
    for key, value in data_dict.items():
        if key == 'Epoch':
            continue
        plt.plot(epoch, value, '.-')
        plt.xticks(epoch, ['{:.0f}'.format(val) for val in epoch])
        plt.title('{} vs Epoch'.format(key))
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.show()
