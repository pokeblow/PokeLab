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
import yaml


class BaseTrainModule():
    def __init__(self, config_path=''):
        '''
            __init__ defines the networks.
        '''
        self.device = None
        # log_value = {'name': '', 'obj': '', 'category': '', 'value_list': [] }
        self.log_lr_dict = []
        self.log_loss_dict = {'train': [], 'valid': []}

        self.model_save_list = []
        self.local_best_loss = []
        self.current_epoch = 0

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def set_lr_log(self, name, obj):
        log_value = {'name': name, 'obj': obj}
        self.log_lr_dict.append(log_value)

    def set_loss_log(self, name, mode=''):
        log_value = {'name': name, 'recent_loss': float('inf'), 'best_loss': float('inf'), 'tmp_list': []}
        self.log_loss_dict[mode].append(log_value)

    def _set_lr(self):
        lr_str_dict = {}
        for i in self.log_lr_dict:
            lr_value = i['obj'].state_dict()['param_groups'][0]['lr']
            lr_str_dict[i['name']] = lr_value

        return lr_str_dict

    def _set_step_loss(self, loss_list=[], mode=''):
        loss_str_dict = {}
        for loss_idx in range(len(loss_list)):
            loss_value = loss_list[loss_idx].item()
            self.log_loss_dict[mode][loss_idx]['tmp_list'].append(loss_value)
            # print out str
            loss_str_dict[self.log_loss_dict[mode][loss_idx]['name']] = loss_value

        return loss_str_dict

    def _update_epoch_loss(self, mode=''):
        loss_str_dict = {}
        for loss_dict in self.log_loss_dict[mode]:
            loss_mean = np.mean(loss_dict['tmp_list'])
            loss_dict['tmp_list'] = []
            loss_dict['recent_loss'] = loss_mean
            if loss_mean < loss_dict['best_loss']:
                loss_dict['best_loss'] = loss_mean
            loss_str_dict[loss_dict['name']] = loss_mean
        return loss_str_dict


    def log_image(self, log_root, current_epoch, log_image_list):
        for image_idx, image_box in enumerate(log_image_list):
            fig = self.show_single_log_image(image_box)
            save_path = '{}/epoch_{}_{}.png'.format(log_root, current_epoch+1, image_idx)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print('log image saved to {}'.format(save_path))
        self.log_image_list = []

    def set_save_parameter(self, model, loss_name, save_path):
        model_save_dict = {'model': model, 'loss_name': loss_name, 'save_path': save_path}
        self.model_save_list.append(model_save_dict)

    def save_parameter(self, epoch, checkpoint_length):
        if self.local_best_loss == []:
            self.local_best_loss = [float('inf') for i in range(len(self.model_save_list))]
        for idx, model_save_dict in enumerate(self.model_save_list):
            for loss_dict in self.log_loss_dict['train']:
                if model_save_dict['loss_name'] == loss_dict['name']:

                    if loss_dict['best_loss'] == loss_dict['recent_loss']:
                        net = model_save_dict['model']
                        if isinstance(net, torch.nn.DataParallel):
                            torch.save(net.module.state_dict(), model_save_dict['save_path'])
                        else:
                            torch.save(net.state_dict(), model_save_dict['save_path'])

                    if checkpoint_length != False:
                        if self.local_best_loss[idx] > loss_dict['recent_loss']:
                            self.local_best_loss[idx] = loss_dict['recent_loss']
                            net = model_save_dict['model']
                            if isinstance(net, torch.nn.DataParallel):
                                torch.save(net.module.state_dict(),
                                           model_save_dict['save_path'][:-4] + '_checkpoint_{}.pth'.format(
                                               str((epoch // checkpoint_length + 1) * checkpoint_length)))
                            else:
                                torch.save(net.state_dict(),
                                           model_save_dict['save_path'][:-4] + '_checkpoint_{}.pth'.format(
                                               str((epoch // checkpoint_length + 1) * checkpoint_length)))

                        if epoch % checkpoint_length == 0:
                            self.local_best_loss = [float('inf') for i in range(len(self.model_save_list))]



    def get_trainer_config(self):
        return self.config

    def set_scheduler(self):
        '''
        set_scheduler defines set scheduler.
        '''

    def show_single_log_image(self):
        '''
        show_single_log_image defines set scheduler.
        '''

    def train_step(self, batch_idx, batch):
        '''
        training_step defines the train loop and backward pipline.
        :param batch: batch data in each loop.
        :param batch: batch index in each epoch.
        :return: value of loss function
        '''

    def valid_step(self, batch_idx, batch):
        '''
        training_step defines the train loop and backward pipline.
        :param batch: batch data in each loop.
        :param batch: batch index in each epoch.
        :return: value of loss function
        '''

    def configure_logs(self):
        '''
        configure_logs defines the logs for loss, acc and lr carves.
        :return:
        '''

    def configure_save_process(self):
        '''
        configure_save defines the models saving process
        use self.set_save_parameter(model, loss_name, save_path, mode='best')
        :return:
        '''

    def show_single_log_image(self, image_box):
        '''
        show the log image with plt
        :param image_box: image box with one set of result image
        :return: None
        '''
        return None
