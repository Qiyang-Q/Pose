import numpy as np
import open3d as o3d
import time
import math
import sys
from tqdm import tqdm as _tqdm
import torch
import os
from skimage import measure
from torch.utils.data import DataLoader
from src.models.ibrnet.sample_ray import RaySamplerSingleImage
EPS = 1e-8


def vis_voxel(voxel):
    vis_voxel = voxel
    pcd_voxel = o3d.geometry.PointCloud()
    pcd_voxel.points = o3d.utility.Vector3dVector(vis_voxel)
    pcd_voxel.paint_uniform_color([0.5, 0.5, 0.5])
    # # o3d.visualization.draw_geometries([pcd_voxel])
    # size = 50
    # bounds = np.array([[0, 0, 0], [0, 0, size], [0, size, 0], [0, size, size], [size, 0, 0], [size, 0, size], [size, size, 0], [size, size, size]])
    # lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    # colors = [[1, 0, 0] for i in range(len(lines))]
    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(bounds)
    # line_set.lines = o3d.utility.Vector2iVector(lines)
    # line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_voxel])


if __name__ == '__main__':
    root_dir = os.getcwd()
    data_dir = '../data/onepose_datasets/train_data'
    # data_dir = '../data/onepose_datasets/val_data'
    # data_dir = '../data/onepose_datasets/test_data'

    all_catagory = {}
    for obj_name in os.listdir(data_dir):
        obj_id = obj_name.split('-')[0]
        all_catagory[obj_id] = obj_name

    tar_obj_id_list = ['0410', '0413', '0414', '0415', '0416', '0418', '0420', '0421', '0443', '0445', '0448', '0460',
                       '0461', '0462', '0463', '0464', '0465', '0477', '0479', '0484', '0499', '0506', '0507', '0509',
                       '0512', '0513', '0516', '0529', '0530', '0531', '0532', '0533', '0536', '0542', '0545', '0546',
                       '0549', '0556', '0561', '0562', '0563', '0566', '0567', '0569', '0571', '0572', '0573', '0574',
                       '0575']
    # tar_obj_id_list = ['0408', '0409', '0419', '0422', '0423', '0424', '0447', '0450', '0452', '0455', '0456', '0458',
    #                    '0459', '0466', '0468', '0469', '0470', '0471', '0472', '0473', '0474', '0476', '0480', '0483',
    #                    '0486', '0487', '0488', '0489', '0490', '0492', '0493', '0494', '0495', '0496', '0497', '0498',
    #                    '0500', '0501', '0502', '0503', '0504', '0508', '0510', '0511', '0517', '0518', '0519', '0520',
    #                    '0521', '0522', '0523', '0525', '0526', '0527', '0534', '0535', '0537', '0539', '0543', '0547',
    #                    '0548', '0550', '0551', '0552', '0557', '0558', '0559', '0560', '0564', '0565', '0568', '0570',
    #                    '0577', '0578', '0579', '0580', '0582', '0583', '0594', '0595']
    tar_obj_id_list = ['0573']
    for tar_obj_id in tar_obj_id_list:
        print('visualizing {}'.format(all_catagory[tar_obj_id]))
        voxel_path = os.path.join(data_dir, all_catagory[tar_obj_id])
        voxel_pos = np.load(os.path.join(voxel_path, 'voxel_position_v2.npy'))
        voxel_crd = np.load(os.path.join(voxel_path, 'correspondence_v2.npy'), allow_pickle=True).item()
        # idx = np.argmin(np.array([len(voxel_crd[i]) for i in voxel_crd.keys()]))
        # for i in voxel_crd.keys():
        #     vis_voxel(voxel_crd[i][:, :3])
        # idx = 129
        # surface = voxel_crd[idx][:, :3]
        # vis_voxel(surface)
        vis_voxel(voxel_pos)



