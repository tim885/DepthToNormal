# convert iBims-1 point2plane depth to point cloud x,y,z
import argparse
import os
import numpy as np
import pandas as pd
import cv2
from math import atan, tan, pi
from tqdm import tqdm
import itertools


img_ext = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']


def zero_depth_mask(depth_path):
    depth = cv2.imread(depth_path, -1)
    mask = (depth == 0).astype(np.uint8)
    mask *= 255
    return mask


# reference:
# https://elcharolin.wordpress.com/2017/09/06/transforming-a-depth-map-into-a-3d-point-cloud/
def create_point_cloud(depth_path, fov_x, fov_y, ratio):
    depth = cv2.imread(depth_path, -1)
    H, W = depth.shape
    point_cloud = []

    # change value to meters
    depth = depth.astype('float64')
    depth *= ratio

    for i, j in itertools.product(range(H), range(W)):
        alpha = (pi - fov_x) / 2
        gamma = alpha + fov_x * float((W - j) / W)
        delta_x = depth[i, j] / tan(gamma)

        alpha = (pi - fov_y) / 2
        gamma = alpha + fov_y * float((H - i) / H)
        delta_y = depth[i, j] / tan(gamma)

        point_cloud.append([delta_x, delta_y, float(depth[i, j])])

    return np.array(point_cloud)


parser = argparse.ArgumentParser(description='Transform depth maps to point cloud given the camera intrinsics')
parser.add_argument('--depth', type=str, default=None, help='path to folder saving all the depth map')
parser.add_argument('--calib', type=str, default=None, help='path to folder saving all the calibration matrix')
parser.add_argument('--pc', type=str, default=None, help='path to save all the point cloud')
parser.add_argument('--mask', type=str, default=None, help='path to save all the masks for invalid depth')
opt = parser.parse_args()


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_name = 'SceneNetRGBD'
    dataset_dir = os.path.join(curr_dir, 'data', dataset_name)
    depth_dir = 'depth_plane'

    scene_dir = os.path.join(dataset_dir, 'train', '0')
    # sel_scene_list = ['900']
    sel_scene_list = list(range(900, 1000))  # scenes for val

    # dataset properties
    depth_ratio = 65.535 / 65535  # meter / array value
    fov_x, fov_y = 60. / 180 * pi, 45. / 180 * pi

    for scene_id, scene_name in enumerate(tqdm(sel_scene_list)):
        curr_scene_dir = os.path.join(scene_dir, str(scene_name))
        curr_depth_dir = os.path.join(curr_scene_dir, depth_dir)
        pc_out_dir = os.path.join(curr_scene_dir, 'pointcloud_xyz')
        mask_out_dir = os.path.join(curr_scene_dir, 'invalid_mask')
        if not os.path.exists(pc_out_dir): os.makedirs(pc_out_dir)
        if not os.path.exists(mask_out_dir): os.makedirs(mask_out_dir)

        file_ids = list(range(0, 6000, 300))
        for idx, file_id in enumerate(file_ids):
            depth_path = os.path.join(curr_depth_dir, '{}.png'.format(file_id))
            pc_out_path = os.path.join(pc_out_dir, '{}.xyz'.format(file_id))
            mask_out_path = os.path.join(mask_out_dir, '{}.png'.format(file_id))

            # create the zero depth mask and save it
            invalid_mask = zero_depth_mask(depth_path)
            cv2.imwrite(mask_out_path, invalid_mask)

            # create the point cloud and save it
            point_cloud = create_point_cloud(depth_path, fov_x, fov_y, ratio=depth_ratio)
            np.savetxt(pc_out_path, point_cloud, fmt='%.3f', newline="\r\n")

