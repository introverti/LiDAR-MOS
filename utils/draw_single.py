import copy
import os
import numpy as np
import pandas as pd
import cv2
import math
from matplotlib import pyplot as plt

MODEL_ADDR = "/home/xavier/deeplearning/LiDAR-MOS/LiDAR-MOS-WAYMO/data/mynet/logs/2022-6-09-18:50"

if __name__ == "__main__":
    loss_addr = os.path.join(MODEL_ADDR, "loss.txt")
    lr_addr = os.path.join(MODEL_ADDR, "lr.txt")

    loss = np.loadtxt(loss_addr)
    lr = np.loadtxt(lr_addr)

    plt.title("SalsaNext on Waymo", fontsize=20)
    plt.xlabel('Batch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.plot(loss, 'r-', linewidth=1)
    # plt.plot(mtest[:-1, 0], mtest[:-1, 2], 'b-', linewidth=1)
    # plt.plot(mtest[:-1, 0], mtest[:-1, 3], 'g-', linewidth=1)
    plt.legend(["Loss"], fontsize=20)
    plt.show()
