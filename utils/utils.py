#!/usr/bin/env python3
# Developed by Tianyun Xuan
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generates Kitti format dataset from Waymo dataset

import os
import tensorflow as tf
import math
import numpy as np

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
    """Convert segmentation labels from range images to point clouds.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      segmentation_labels: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      ri_index: 0 for the first return, 1 for the second return.

    Returns:
      point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(
        frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(
                tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(
                sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(
                tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels


def get_pointcloud(frame, lidar_name=0):
    (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
        frame)
    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)
    point_labels = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels)
    # 3d points in vehicle frame.
    # points_all = np.concatenate(points, axis=0)
    if lidar_name < 0 or lidar_name > 4:
        lidar_name = 0
    return points[lidar_name], point_labels[lidar_name]


def get_pose(frame):
    # Input : one single frame data
    # Output : Pose and pointcloud of selected lidar_name
    # Frame vehicle pose. Note that unlike in CameraImage, the Frame pose does
    # not correspond to the provided timestamp (timestamp_micros). Instead, it
    # roughly (but not exactly) corresponds to the vehicle pose in the middle of
    # the given frame. The frame vehicle pose defines the coordinate system which
    # the 3D laser labels are defined in.
    pose = np.reshape(np.array(frame.pose.transform),
                      [4, 4]).flatten().tolist()
    return pose


def get_timestamp(frame):
    # 16 digitals microseconds
    return frame.timestamp_micros

# def get_label(frame):
#     return frame.laser_labels


def parse_tfrecord(file_name):
    # parse single tfrecord and return frames in array
    dataset = tf.data.TFRecordDataset(file_name, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        yield frame


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def exist(path):
    return os.path.exists(path)


def load_files(folder):
    """ Load all files in a folder and sort.
    """
    file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(folder)) for f in fn]
    file_paths.sort()
    return file_paths


def load_vertex(scan_path):
    """ Load 3D points of a scan. The fileformat is the .npy format used in
      the inno dataset.
      Args:
        scan_path: the (full) filename of the scan file
      Returns:
        A nx4 numpy array of homogeneous points (x, y, z, 1).
    """
    current_points = np.load(scan_path)
    current_vertex = np.ones(
        (current_points.shape[0], current_points.shape[1] + 1))
    current_vertex[:, :-1] = current_points
    return current_vertex


def load_label(label_path):
    full_label = np.load(label_path)
    instance = full_label[:, 0]
    sematic = full_label[:, 1]
    return instance , sematic


def range_projection(current_vertex, proj_H=64, proj_W=900, fov_up=3.0, fov_down=-25.0, max_range=50, min_range=2):
    """ Project a pointcloud into a spherical projection, range image.
      Args:
        current_vertex: raw point clouds (x, y, z, 1)
      Returns:
        proj_vertex: each pixel contains the corresponding point (x, y, z, depth)
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    current_vertex = current_vertex[(depth > min_range) & (
        depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > min_range) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    intensity = current_vertex[:, 3]

    # get angles of all points
    # x for roll  (front positive)
    # y for pitch (left positive)
    # z for yaw   (up positive)
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    proj_x_orig = np.copy(proj_x)

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    proj_y_orig = np.copy(proj_y)

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                       dtype=np.int32)  # [H,W] index (-1 is no data)
    proj_intensity = np.full((proj_H, proj_W), -1,
                             dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, depth]).T
    proj_idx[proj_y, proj_x] = indices
    proj_intensity[proj_y, proj_x] = intensity,
    return proj_vertex

def trace(*txt):
    for item in txt:
        print("\033[0;37;41m\t" + str(txt) + "\033[0m")