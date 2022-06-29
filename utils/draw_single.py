import copy
import os
import numpy as np
import pandas as pd
import cv2
import math
from matplotlib import pyplot as plt

MODEL_ADDR = "/home/xavier/deeplearning/LiDAR-MOS/LiDAR-MOS-WAYMO/data/mynet/logs/2022-6-21-11:24/analyse"

if __name__ == "__main__":
    jacc_addr = os.path.join(MODEL_ADDR, "jacc.txt")
    wcs_addr = os.path.join(MODEL_ADDR, "wcs.txt")
    # lr_addr = os.path.join(MODEL_ADDR, "lr.txt")

    jacc = np.loadtxt(jacc_addr)
    wcs = np.loadtxt(wcs_addr)
    # lr = np.loadtxt(lr_addr)


    plt.title("SalsaNext on Waymo", fontsize=20)
    plt.xlabel('Batch', fontsize=20)
    plt.ylabel('Lovasz_softmax', fontsize=20)
    #plt.ylabel('NLLLoss', fontsize=20)
    # plt.plot(jacc, 'r-', linewidth=1)
    plt.plot(wcs, 'b-', linewidth=1)
    # plt.plot(lr, 'g-', linewidth=1)
    plt.legend(["Lovasz_softmax"], fontsize=20)
    plt.show()
