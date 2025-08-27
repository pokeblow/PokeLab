import torch
import platform
import re
import numpy as np
import shutil
import os
import logging

class LogModule():
    def __init__(self, log_name):
        self.log_name = log_name

        self.c_epoch = 0
        self.c_mean = 0

        self.tmp_list = []

    def set_value(self, value):
        self.tmp_list.append(value)

    def get_epoch_summary(self):
        self.c_mean = np.mean(self.tmp_list)
        self.tmp_list = []
        self.c_epoch += 1

        return self.log_name, self.c_mean

if __name__ == "__main__":
    myLogs = Logs(globals.LOG_ROOT_PATH)
    test_log1 = LogModule('loss1')
    test_log2 = LogModule('loss2')
    myLogs.set_log_module(test_log1, test_log2)





