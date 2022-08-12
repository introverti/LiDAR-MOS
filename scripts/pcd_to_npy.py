import numpy as np
import os
import open3d as o3d
from tqdm import tqdm

EXTENSIONS_PCD = [".pcd"]
if __name__ == '__main__':
    PCD_ADDR = "/home/xavier/repos/floam/data/lidar"
    BIN_ADDR = "/home/xavier/repos/LiDAR-MOS-WAYMO/data/falcon/sequences/01/falcon"
    pcds = os.listdir(PCD_ADDR)
    pcds.sort()
    for pcd_idx in tqdm(range(len(pcds))):
        pcd = pcds[pcd_idx]
        if (pcd.endswith(ext) for ext in EXTENSIONS_PCD):
            pc = o3d.io.read_point_cloud(os.path.join(PCD_ADDR, pcd))
            xyz = np.array(pc.points)
            to_lidar_coor = np.array([0,0,1,0,-1,0,1,0,0]).reshape(3,3)
            zmyx = xyz@to_lidar_coor
            np.save(os.path.join(BIN_ADDR, pcd.replace(".pcd", "")), zmyx)
