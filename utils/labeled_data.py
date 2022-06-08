import os
import sys
import numpy as np
from tqdm import tqdm
from utils import *
import shutil
import yaml

WAYMO_DATASET_ADDR = "/home/xavier/deeplearning/LiDAR-MOS/xaiver/data/own/sequences"
LABELED_DATASET_ADDR = "/home/xavier/deeplearning/LiDAR-MOS/xaiver/data/labeled/sequences"

if __name__ == '__main__':
    assert(exist(WAYMO_DATASET_ADDR))
    sequences = os.listdir(WAYMO_DATASET_ADDR)
    sequences.sort()
    max_points = 0
    for index in tqdm(range(len(sequences))):
        scene_addr = os.path.join(
                WAYMO_DATASET_ADDR, str(index).zfill(2))
        
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

        labels = os.listdir(label_addr)
        count = 0

        save_addr = os.path.join(LABELED_DATASET_ADDR, str(index).zfill(2))
        save_lidar = os.path.join(save_addr, "lidars")
        save_label = os.path.join(save_addr, "labels")
        save_residual = os.path.join(save_addr, "residual_images_1")
        save_pose_file = os.path.join(save_addr, "poses.npy")

        check_path(save_addr)
        check_path(save_lidar)
        check_path(save_label)
        check_path(save_residual)

        new_poses = []
        for id in range(len(labels)):
            idx = str(id).zfill(6)+".npy"
            count_idx = str(count).zfill(6)+".npy"

            label_frame = os.path.join(label_addr, idx)
            label_array = np.load(label_frame).reshape(-1,2)
            max_points =max(max_points, label_array.shape[0])
            if not ((label_array == [0,0]).all()):
                new_poses.append(poses[id])

                lidar_frame = os.path.join(lidar_addr, idx)
                residual_frame = os.path.join(residual_addr, idx)

                new_label_frame = os.path.join(save_label, count_idx)
                new_lidar_frame = os.path.join(save_lidar, count_idx)
                new_residual_frame = os.path.join(save_residual, count_idx)

                shutil.copy(label_frame, new_label_frame)
                shutil.copy(lidar_frame, new_lidar_frame)
                shutil.copy(residual_frame, new_residual_frame)
                count += 1
            
            assert(count == len(new_poses))
            np.save(save_pose_file, new_poses)
    print ("Maximum points size : ", max_points)



