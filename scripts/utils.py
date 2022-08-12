#!/usr/bin/env python3
# Developed by Tianyun Xuan
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generates Kitti format dataset from Waymo dataset

import os
import tensorflow as tf
import math
import numpy as np


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


def range_projection(current_vertex, proj_H=64, proj_W=900, fov_up=3.0, fov_down=-25.0, fov_horizontal=120, max_range=50, min_range=2):
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
    fov_h = fov_horizontal/ 180.0 *np.pi
    demi_fov_h = fov_h /2

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

    # [-pi, pi] -> [-fov_h/2, fov_h/2]
    # get projections in image coords
    # proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_x = (yaw + demi_fov_h) / fov_h
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
