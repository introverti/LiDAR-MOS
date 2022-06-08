import os
import sys
import numpy as np
from tqdm import tqdm
from utils import *
import yaml


if __name__ == '__main__':
    # load config file
    config_filename = 'config/data_transform.yaml'
    if len(sys.argv) > 1:
        config_filename = sys.argv[1]

    if yaml.__version__ >= '5.1':
        config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(config_filename))
    
    output_folder = config['Kitti_format_data_folder']
    scenes = os.listdir(output_folder)
    scenes.sort()

    # need to vote
    ratio = np.zeros((1,23))
    img_means = np.zeros((1,4), dtype=np.float64) #range,x,y,z (signal?)
    tr_means = np.array([18.54111379, 1.74578001, 0.87203252, 1.1462703], dtype=np.float64).reshape((1,4))
    img_stds = np.zeros((1,4), dtype=np.float64)
    assert (tr_means.shape == (1,4))
    point_num = 0
    labeled_point = 0

    for scene in scenes:
        scene_folder = os.path.join(output_folder, scene)

        # Lidars
        scans_folder = os.path.join(scene_folder, "lidars")
        if not exist(scans_folder):
            print("Cannot find lidar scan pc file", scans_folder)
            continue
        scans_paths = load_files(scans_folder)
        scans_paths.sort()

        # Lables
        labels_folder = os.path.join(scene_folder, "labels")
        if not exist(labels_folder):
            print("Cant find labels folder")
            continue
        labels_paths = load_files(labels_folder)
        labels_paths.sort()

        # assert(len(scans_paths)==len(labels_path))

        for frame_idx in tqdm(range(len(scans_paths))):
            _, sematic = load_label(labels_paths[frame_idx])
            if not ((sematic == 0).all()):
                labeled_point += sematic.shape[0]
                for label in sematic:
                    ratio[0,label]+=1

        #     current_scan = load_vertex(scans_paths[frame_idx])
        #     range_norm =np.linalg.norm(current_scan, 2, axis=1, keepdims=True)
        #     temp_res = np.concatenate([range_norm, current_scan[:, :3]], axis=1)

        #     temp_sum = np.sum(temp_res, axis = 0, keepdims=True)
        #     assert (temp_sum.shape == (1,4))
        #     img_means += temp_sum

        #     temp_diff = temp_res - tr_means
        #     temp_diff_carr = np.power(temp_diff, 2)
        #     temp_diff_carr_sum = np.sum(temp_diff_carr, axis=0, keepdims=True)
        #     img_stds += temp_diff_carr_sum

        #     point_num += current_scan.shape[0]

    # print ("Point number : ", point_num)
    # print ("Means : ", img_means/point_num)
    # print ("Stds : ", np.sqrt(img_stds/point_num))
    print ("Labeled point : " , labeled_point)
    print ("Ratio : ", ratio/labeled_point)