from typing import Dict, Tuple, Iterable, Optional
import os
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime
import shutil
import logging
import platform
from torchvision import transforms
from PIL import Image

import torch

time = datetime.now()
DATE_TIME = time.strftime('%Y%m%d%H')


class Evaluation:
    def __init__(self, results_path=''):
        self.results_path = results_path
        self.result_box = {}

    def configure_metrics(self):
        pass
    def evaluate_step(self, result_box):
        raise NotImplementedError

    def _input_results(self, path):
        pass


    def evaluate(self):
        result_box = self._input_results(self.results_path)
        self.evaluate_step(result_box)


class MultiEvaluation:
    def __init__(self, results_dict={}):
        self.results_dict = results_dict

    def evaluate_step(self, result_box):
        raise NotImplementedError

    def _input_results(self, path):
        pass

    def evaluate(self):
        for item_name, results_path in self.results_dict.items():
            result_box = self._input_results(results_path)
            self.evaluate_step(result_box)

