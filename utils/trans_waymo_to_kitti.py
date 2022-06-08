#!/usr/bin/env python3
# Developed by Tianyun Xuan
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generates Kitti format dataset from Waymo dataset

import os
import sys
import yaml
import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm
import itertools

from utils import *
import open3d as o3d

if __name__ == '__main__':
    # load config file
    config_filename = 'config/data_transform.yaml'
    if len(sys.argv) > 1:
        config_filename = sys.argv[1]
    if yaml.__version__ >= '5.1':
        config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(config_filename))

    # specify parameters
    data_type = config["type"]
    debug = config['debug']
    num_frames = config['num_frames']
    normalize = config['normalize']
    num_last_n = config['num_last_n']

    base_folder = config['source_folder']
    lidar_name = config['lidar_name']

    output_folder = config['Kitti_format_data_folder']
    visualize = config['visualize']

    range_image_params = config['range_image']
    check_path(output_folder)

    scenes = os.listdir(base_folder)
    count = 0
    for record_id in tqdm(range(len(scenes))):
        # sequences path
        scenes_folder = os.path.join(output_folder, str(count).zfill(2))
        check_path(scenes_folder)
        labels_folder = os.path.join(scenes_folder, "labels")
        check_path(labels_folder)
        lidar_folder = os.path.join(scenes_folder, "lidars")
        check_path(lidar_folder)

        record_path = os.path.join(base_folder, scenes[record_id])
        frame_idx = 0
        poses = []
        # timestamps = []
        for frame in parse_tfrecord(record_path):
            # print("Sequence: ", count, " Frame: ", frame_idx)
            poses.append(get_pose(frame))
            # timestamps.append(get_timestamp(frame))

            pointcloud, label = get_pointcloud(frame, lidar_name)
            if pointcloud.shape[0] == label.shape[0]:
                print("Sequence: ", count, " Frame: ", frame_idx)
                pc_name = os.path.join(lidar_folder, str(frame_idx).zfill(6))
                np.save(pc_name, pointcloud)
                # print ("<<< Saved PC shape : ", pointcloud.shape)

                label_name = os.path.join(labels_folder, str(frame_idx).zfill(6))
                np.save(label_name, label)
                # print ("<<< Saved label shape : ", label.shape)
            else :
                print ("PC shape doesnot match with label")

            frame_idx += 1
        if frame_idx == len(poses):
            pose_name = os.path.join(scenes_folder,"poses")
            np.save(pose_name, poses)
        else :
            print ("Poses shape doesnot match with frame number")

        # timestamp_name = os.path.join(scenes_folder,"times.txt")
        # np.savetxt(timestamp_name, timestamps)
        count += 1
