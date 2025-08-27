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
from .globals import *
import shutil
import yaml
from tqdm import tqdm


class Trainer():
    def __init__(self, train_module=None, train_dataset=None, valid_dataset=None):
        self.device = train_module.device
        self.version = ''

        self.train_module = train_module
        self.train_loader = None
        self.valid_loader = None
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.epochs = 1
        self.batch_size = 1
        self.result_number = 0

        self.log_root = LOG_ROOT_PATH
        print(self.log_root)

        if os.path.exists(self.log_root):
            shutil.rmtree(self.log_root)
        os.makedirs(self.log_root, exist_ok=True)
        log_file = os.path.join(self.log_root, "training_summary.log")

        with open(log_file, 'w'):
            pass

        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger = logging.getLogger('Logger')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

    def __set_Dataset(self, train_dataset=None, valid_dataset=None):
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        drop_last=True
                                                        )
        self.valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        drop_last=True
                                                        )

    def configure(self):
        config = self.train_module.get_trainer_config()

        self.train_module.configure_logs()
        self.train_module.configure_save_process()

        self.version = config['version']
        self.check_point = config['train']['check_point']
        self.epochs = config['train']['epochs']
        self.batch_size = config['train']['batch_size']
        self.show_step_loss = config['train']['show_step_loss']

        self.__set_Dataset(self.train_dataset, self.valid_dataset)

    def __log_step(self, epoch, batch_idx, steps, loss_str_dict, lr_str_dict):
        log_message = (f"Epoch: {epoch + 1}, Step: {batch_idx + 1} / {steps}, "
                       f"lr: {', '.join([f'{key}: {value:.6f}' for key, value in lr_str_dict.items()])}, "
                       f"Losses: {', '.join([f'{key}: {value:.6f}' for key, value in loss_str_dict.items()])}")
        print(log_message)

    def __log_epoch_summary(self, epoch, str_lr_dict, epoch_loss_dict_train, epoch_loss_dict_valid):
        log_message = (f"Epoch: {epoch + 1}, "
                       f"lr: {', '.join([f'{key}: {value:.6f}' for key, value in str_lr_dict.items()])}, "
                       f"Losses: {', '.join([f'{key}: {value:.6f}' for key, value in epoch_loss_dict_train.items()])}"
                       f", {', '.join([f'{key}: {value:.6f}' for key, value in epoch_loss_dict_valid.items()])}")
        self.logger.info(log_message)
        print_message = (f"Epoch: {epoch + 1}\n"
                         f"lr: {', '.join([f'{key}: {value:.6f}' for key, value in str_lr_dict.items()])}\n"
                         f"Train: {', '.join([f'{key}: {value:.6f}' for key, value in epoch_loss_dict_train.items()])}\n"
                         f"Valid: {', '.join([f'{key}: {value:.6f}' for key, value in epoch_loss_dict_valid.items()])}")
        print(print_message)

    def fit(self):
        self.logger.info(f'Version: {self.version}')
        self.logger.info(f'batch_size={self.batch_size}, '
                         f'num_epochs={self.epochs}, device={self.device}')

        self.logger.info(f"train dataset: {len(self.train_dataset)}, valid dataset: {len(self.valid_dataset)}")

        str_lr_dict = self.train_module._set_lr()
        for epoch in range(self.epochs):

            for idx, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch: {epoch + 1}, train step')):
                batch = [tensor.to(device=self.device, dtype=torch.float32) for tensor in batch]
                loss = self.train_module.train_step(idx, batch)
                if not isinstance(loss, tuple):
                    loss = (loss,)
                log_loss_str_dict = self.train_module._set_step_loss(loss_list=loss, mode='train')
                if self.show_step_loss:
                    self.__log_step(epoch, idx, len(self.train_loader), str_lr_dict, log_loss_str_dict)

            epoch_loss_dict_train = self.train_module._update_epoch_loss(mode='train')

            log_image_list = []
            with torch.no_grad():
                for idx, batch in enumerate(
                        tqdm(self.valid_loader, desc='{}  valid step'.format(' ' * len(f'Epoch: {epoch + 1}')))):
                    batch = [tensor.to(device=self.device, dtype=torch.float32) for tensor in batch]
                    loss_batch = self.train_module.valid_step(idx, batch)
                    loss = loss_batch[:-1]
                    image_list = loss_batch[-1]

                    if self.device == 'cpu':
                        image_list = [tensor.detach().numpy()[0] for tensor in image_list]
                    else:
                        image_list = [tensor.cpu().detach().numpy()[0] for tensor in image_list]

                    if len(image_list) != 0:
                        log_image_list.append(image_list)

                    log_loss_str_dict = self.train_module._set_step_loss(loss_list=loss, mode='valid')
                    if self.show_step_loss:
                        self.__log_step(epoch, idx, len(self.valid_loader), str_lr_dict, log_loss_str_dict)

                epoch_loss_dict_valid = self.train_module._update_epoch_loss(mode='valid')

            self.__log_epoch_summary(epoch, str_lr_dict, epoch_loss_dict_train, epoch_loss_dict_valid)
            self.train_module.log_image(self.log_root, epoch, log_image_list)

            self.train_module.save_parameter(epoch, self.check_point)

            self.train_module.set_scheduler()

    def fit_valid(self):
        self.logger.info(f'Version: {self.version}')
        self.logger.info(f'batch_size={self.batch_size}, '
                         f'num_epochs={self.epochs}, device={self.device}')

        self.logger.info(f"train dataset: {len(self.train_dataset)}, valid dataset: {len(self.valid_dataset)}")

        log_image_list = []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.valid_loader, desc='valid step')):
                batch = [tensor.to(device=self.device, dtype=torch.float32) for tensor in batch]
                loss_batch = self.train_module.valid_step(idx, batch)
                loss = loss_batch[:-1]
                image_list = loss_batch[-1]

                if self.device == 'cpu':
                    image_list = [tensor.detach().numpy()[0] for tensor in image_list]
                else:
                    image_list = [tensor.cpu().detach().numpy()[0] for tensor in image_list]

                if len(image_list) != 0:
                    log_image_list.append(image_list)

        self.train_module.log_image(self.log_root, -1, log_image_list)
