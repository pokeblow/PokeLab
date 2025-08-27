import re
import matplotlib.pyplot as plt
from collections import defaultdict

import re
import matplotlib.pyplot as plt
from collections import defaultdict
import math

def monitor(log_path):
    # 初始化存储结构
    epochs = []
    lr_dict = defaultdict(list)
    loss_train = defaultdict(list)
    loss_val = defaultdict(list)

    # 正则表达式
    epoch_re = re.compile(r"Epoch:\s*(\d+)")
    lr_re = re.compile(r"\b(lr_[A-Za-z0-9_]+):\s*([\d.eE+-]+)")
    loss_re = re.compile(r"\b(Loss_[A-Za-z0-9_]+(?:_val)?):\s*([\d.eE+-]+)")

    # 读取日志文件
    with open(log_path, 'r') as f:
        for line in f:
            if 'Epoch:' not in line:
                continue

            # 解析 epoch
            epoch_match = epoch_re.search(line)
            if not epoch_match:
                continue
            epoch = int(epoch_match.group(1))
            epochs.append(epoch)

            # 提取学习率
            for lr_name, lr_value in lr_re.findall(line):
                lr_dict[lr_name].append(float(lr_value))

            # 提取损失值
            for loss_name, loss_value in loss_re.findall(line):
                val = float(loss_value)
                if loss_name.endswith('_val'):
                    key = loss_name[:-4]  # 去掉 _val 后缀作为统一 key
                    loss_val[key].append(val)
                else:
                    loss_train[loss_name].append(val)

    # 所有损失项名称
    all_loss_keys = sorted(set(loss_train.keys()).union(loss_val.keys()))
    num_losses = len(all_loss_keys)

    # ---------- 绘制学习率图 ----------
    plt.figure(figsize=(10, 4))
    for name, values in lr_dict.items():
        plt.plot(epochs, values, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rates Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- 绘制每个loss的子图 ----------
    fig, axes = plt.subplots(num_losses, 1, figsize=(10, 4 * num_losses), sharex=True)
    if num_losses == 1:
        axes = [axes]  # 保证迭代性

    for i, key in enumerate(all_loss_keys):
        if key in loss_train:
            axes[i].plot(epochs, loss_train[key], label=f'{key} (train)')
        if key in loss_val:
            axes[i].plot(epochs, loss_val[key], label=f'{key} (val)', linestyle='--')
        axes[i].set_title(f"{key} Loss Over Epochs")
        axes[i].set_ylabel("Loss")
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.show()


    # # 初始化
    # epochs = []
    # lr_dict = defaultdict(list)
    # loss_train = defaultdict(list)
    # loss_val = defaultdict(list)

    # # 正则表达式
    # epoch_re = re.compile(r"Epoch:\s*(\d+)")
    # lr_re = re.compile(r"\b(lr_[A-Za-z0-9_]+):\s*([\d.eE+-]+)")
    # loss_re = re.compile(r"\b(Loss_[A-Za-z0-9_]+(?:_val)?):\s*([\d.eE+-]+)")

    # # 读取日志
    # with open(log_path, 'r') as f:
    #     for line in f:
    #         if 'Epoch:' in line:
    #             epoch_match = epoch_re.search(line)
    #             if not epoch_match:
    #                 continue
    #             epoch = int(epoch_match.group(1))
    #             epochs.append(epoch)

    #             # 提取学习率
    #             for lr_name, lr_value in lr_re.findall(line):
    #                 lr_dict[lr_name].append(float(lr_value))

    #             # 提取损失
    #             for loss_name, loss_value in loss_re.findall(line):
    #                 val = float(loss_value)
    #                 if loss_name.endswith('_val'):
    #                     loss_val[loss_name].append(val)
    #                 else:
    #                     loss_train[loss_name].append(val)

    # # 绘图：学习率
    # plt.figure(figsize=(10, 4))
    # for name, values in lr_dict.items():
    #     plt.plot(epochs, values, label=name)
    # plt.xlabel("Epoch")
    # plt.ylabel("Learning Rate")
    # plt.title("Learning Rates Over Epochs")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # 绘图：训练和验证损失分两个子图
    # fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # # 子图1：训练损失
    # for name, values in loss_train.items():
    #     axes[0].plot(epochs, values, label=name)
    # axes[0].set_title("Training Losses")
    # axes[0].set_ylabel("Loss")
    # axes[0].legend()
    # axes[0].grid(True)

    # # 子图2：验证损失
    # for name, values in loss_val.items():
    #     axes[1].plot(epochs, values, label=name, linestyle='--')
    # axes[1].set_title("Validation Losses")
    # axes[1].set_xlabel("Epoch")
    # axes[1].set_ylabel("Loss")
    # axes[1].legend()
    # axes[1].grid(True)

    # plt.tight_layout()
    # plt.show()

