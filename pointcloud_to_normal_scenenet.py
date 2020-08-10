# generate surface normal from point cloud for scenenet dataset

# Deep Learning for Robust Normal Estimation in Unstructured Point Clouds
# Copyright (c) 2016 Alexande Boulch and Renaud Marlet
#
# This program is free software; you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation;
# either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details. You should have received a copy of
# the GNU General Public License along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
#
# PLEASE ACKNOWLEDGE THE AUTHORS AND PUBLICATION:
# "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
# by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016,
# Computer Graphics Forum
#
# The full license can be retrieved at https://www.gnu.org/licenses/gpl-3.0.en.html


import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
import python.lib.python.NormalEstimatorHough as Estimator


parser = argparse.ArgumentParser(description='Transform depth maps to point cloud given the camera intrinsics')
parser.add_argument('--pc', type=str, default=None, help='path to directory saving all the point clouds')
parser.add_argument('--depth_plane', type=str, default=None, help='path to directory saving the plane-to-plane depth maps')
parser.add_argument('--depth_point', type=str, default=None, help='path to directory saving the point-to-point depth maps')
parser.add_argument('--normal_xyz', type=str, default=None, help='path to save all the normals in .xyz file')
parser.add_argument('--normal_img', type=str, default=None, help='path to save all the normals in .png file')
opt = parser.parse_args()

curr_dir = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'SceneNetRGBD'
dataset_dir = os.path.join(curr_dir, 'data', dataset_name)
depth_dir = 'depth_plane'
pc_dir = 'pointcloud_xyz'
mask_dir = 'invalid_mask'
scene_dir = os.path.join(dataset_dir, 'train', '0')
# sel_scene_list = ['900']
sel_scene_list = list(range(900, 1000))  # scenes for val
for scene_id, scene_name in enumerate(tqdm(sel_scene_list)):
    curr_scene_dir = os.path.join(scene_dir, str(scene_name))
    curr_depth_dir = os.path.join(curr_scene_dir, depth_dir)
    curr_pc_dir = os.path.join(curr_scene_dir, pc_dir)
    curr_mask_dir = os.path.join(curr_scene_dir, mask_dir)
    normal_xyz_out_dir = os.path.join(curr_scene_dir, 'normal_alex_xyz')
    normal_img_out_dir = os.path.join(curr_scene_dir, 'normal_alex')
    if not os.path.exists(normal_xyz_out_dir): os.makedirs(normal_xyz_out_dir)
    if not os.path.exists(normal_img_out_dir): os.makedirs(normal_img_out_dir)

    file_ids = list(range(0, 6000, 300))
    for idx, file_id in enumerate(file_ids):
        depth_path = os.path.join(curr_depth_dir, '{}.png'.format(file_id))
        pc_path = os.path.join(curr_pc_dir, '{}.xyz'.format(file_id))
        normal_xyz_path = os.path.join(normal_xyz_out_dir, '{}.xyz'.format(file_id))
        normal_img_path = os.path.join(normal_img_out_dir, '{}.png'.format(file_id))

        pts = np.loadtxt(pc_path)
        assert pts.shape[-1] == 3, "The last channel of point cloud must be 3 representing X, Y, Z"
        depth = cv2.imread(depth_path, -1)
        H, W = depth.shape

        # create the normal map and save it in .xyz
        estimator = Estimator.NormalEstimatorHough()
        estimator.set_points(pts)
        estimator.set_K(50)
        estimator.estimate_normals()
        estimator.saveXYZ(normal_xyz_path)

        # read in .xyz and transform it to .png of uint-16
        xyz_normal = np.loadtxt(normal_xyz_path)
        normal = xyz_normal[:, 3:].reshape(H, W, 3)
        normal_uint = ((2**16 - 1) * (normal + 1) / 2).astype('uint16')
        cv2.imwrite(normal_img_path, normal_uint[:, :, [2, 1, 0]])
