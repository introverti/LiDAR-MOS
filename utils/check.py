import os
import sys
import numpy as np
from tqdm import tqdm
from utils import *
import yaml

LABELED_DATASET_ADDR = "/home/xavier/deeplearning/LiDAR-MOS/xaiver/data/labeled/sequences"

if __name__ == '__main__':
    assert(exist(LABELED_DATASET_ADDR))
    sequences = os.listdir(LABELED_DATASET_ADDR)
    sequences.sort()
    
    for index in tqdm(range(len(sequences))):
        scene_addr = os.path.join(
                LABELED_DATASET_ADDR, str(index).zfill(2))
        
        label_addr = os.path.join(scene_addr, "labels")
        lidar_addr = os.path.join(scene_addr, "lidars")
        residual_addr = os.path.join(scene_addr, "residual_images_1")
        pose_file = os.path.join(scene_addr, "poses.npy")
        
        assert(exist(scene_addr))
        assert(exist(label_addr))
        assert(exist(lidar_addr))
        assert(exist(residual_addr))
        assert(exist(pose_file))

        poses = np.load(pose_file).reshape(-1,16)
        assert(poses.shape[1]==16)

        if (np.isnan(poses).any()):
            trace(pose_file+ " has nan !")
        
        labels = os.listdir(label_addr)
        
        for id in range(len(labels)):
            idx = str(id).zfill(6)+".npy"

            label_frame = os.path.join(label_addr, idx)
            label_array = np.load(label_frame).reshape(-1,2)
            if (np.isnan(label_array).any()):
                trace(label_frame + " has nan !")

            lidar_frame = os.path.join(lidar_addr, idx)
            lidar_array = np.load(lidar_frame).reshape(-1,3)
            if (np.isnan(lidar_array).any()):
                trace(lidar_frame + " has nan !")

            residual_frame = os.path.join(residual_addr, idx)
            residual_array = np.load(residual_frame)
            if (np.isnan(residual_array).any()):
                trace(residual_frame + " has nan !")



            


