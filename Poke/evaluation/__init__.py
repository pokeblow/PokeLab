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

class Evaluation:
    def __init__(self, save_root: str = "", matrix: dict = None, dataset: Dataset = None, batch_size: int = 1):
        """
        Args:
            save_root (str): 结果保存路径，默认空字符串
            matrix (dict): 初始化指标矩阵，默认空字典
        """
        self.save_root = save_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matrix = matrix if matrix is not None else {}
        self.dataset = dataset if dataset is not None else None
        self.batch_size = batch_size

    def predict_step(self, batch_idx: int, batch):
        '''

        :param batch_idx:
        :param batch:
        :return:
        '''
        raise NotImplementedError

    def evaluate_step(self, result_box):
        raise NotImplementedError

    def _save_images(self, case_name, image_box):
        for i in image_box:
            for batch in range(self.batch_size):
                if i.shape[1] != 1:
                    image = i.detach().cpu().numpy()[batch].transpose((1, 2, 0))
                else:
                    image = i.detach().cpu().numpy()[batch]
                save_path = f'{self.save_root}/{case_name[batch]}'
                Image.fromarray(image).save(save_path)

    def evaluate(self, *args, **kwargs):
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size)

        pbar = tqdm(self.train_loader, desc=f"Evaluation")
        for idx, batch in enumerate(pbar):
            case_name = batch[0]
            batch = batch[1:]
            result_box = self.predict_step(idx, batch)
            self._save_images(case_name, result_box['images'])
            evaluate_step = self.evaluate_step(result_box)






