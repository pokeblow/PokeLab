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
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import torch

time = datetime.now()
DATE_TIME = time.strftime('%Y%m%d%H')


@dataclass
class PokeResult:
    result_name: str = ""
    result_type: str = ""  # 修正拼写
    result_buffer: Optional[Any] = None  # 默认 None，支持任意格式

    def set_step(self, value) -> None:
        """记录一个 step 的数值。"""
        self.result_buffer = value


class Prediction:
    def __init__(self, save_root: str = "", dataset: Dataset = None, batch_size: int = 1):
        """
        Args:
            save_root (str): 结果保存路径，默认空字符串
            matrix (dict): 初始化指标矩阵，默认空字典
        """
        self.save_root = save_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset if dataset is not None else None
        self.batch_size = batch_size
        self.result_box = []

    def configure_results(self):
        """
        用户可重写，返回需要跟踪的日志集合。
        """
        return None

    def configure_model(self):
        raise NotImplementedError

    def predict_step(self, batch_idx: int, batch):
        '''

        :param batch_idx:
        :param batch:
        :return:
        '''
        raise NotImplementedError

    def _save_images(self, case_name, image_result, batch):
        img = image_result.result_buffer
        result_name = image_result.result_name
        if img.shape[1] != 1:
            image = img.detach().cpu().numpy()[batch].transpose((1, 2, 0))
        else:
            image = img.detach().cpu().numpy()[batch]
        image = (image * 255).astype(np.uint8)
        os.makedirs(f'{self.save_root}/{result_name}', exist_ok=True)
        save_path = f'{self.save_root}/{result_name}/{case_name[batch]}.bmp'
        Image.fromarray(image).save(save_path)


    def predict(self):
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size)
        self.configure_model()
        print('Model configure completed')
        self.result_box = self.configure_results()
        print(f'Result configure completed ({len(self.result_box)} results)')
        print(f'Evaluation dataset: {len(self.data_loader) * self.batch_size}')

        if os.path.exists(self.save_root):
            shutil.rmtree(self.save_root)
            print('Save root folder reset')
        os.makedirs(self.save_root)

        save_path = f'{self.save_root}/result_values.txt'
        if not os.path.isfile(save_path):
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"{DATE_TIME}\n")

        with torch.no_grad():
            pbar = tqdm(self.data_loader, desc=f"Evaluation")
            for idx, batch in enumerate(pbar):
                case_name = batch[0]
                batch = batch[1:]
                self.predict_step(idx, batch)

                for batch in range(self.batch_size):
                    value_str = f'{case_name[batch]}, '
                    for idx, item in enumerate(self.result_box):
                        if item.result_type == 'image':
                            self._save_images(case_name, item, batch)
                        if item.result_type == 'value':
                            value_str += f"{item.result_name} = {item.result_buffer.detach().cpu().numpy()[batch]}, "

                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(f"{value_str}\n")

            print(f'Results saved to /{self.save_root}')
            print('Prediction completed')
