import torch
import matplotlib.pyplot as plt
import numpy as np

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Poke.pipeline.utils import tensor_to_numpy


def visualization(image_box):
    image = tensor_to_numpy(image_box)
    fig, ax = plt.subplots()  # 在 figure 上创建一个子图
    ax.imshow(image[0][0], cmap="gray")  # 如果是单通道图像，建议加 cmap

    return fig
