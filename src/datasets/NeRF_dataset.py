import os.path
import cv2
try:
    import ujson as json
except ImportError:
    import json
import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from src.utils import data_utils
import imageio
import os.path as osp
import matplotlib.pyplot as plt


def load_data(anns, coco, res=128, load_in_ram=False):
    data_idx = {}
    catagory_data = {}
    for idx in anns:
        ann_id = coco.getAnnIds(imgIds=idx)
        anno = coco.loadAnns(ann_id)[0]
        catagory_name = anno['pose_file'].split('/')[-4]
        if catagory_name not in catagory_data:
            img_id = 0
            bbox3d_file = anno['pose_file'].split('/')[:-3]
            bbox3d_file = '/'.join(bbox3d_file) + '/box3d_corners.txt'
            radius = data_utils.get_radius(bbox3d_file)
            catagory_data[catagory_name] = {'radius': radius,
                                            'data': {}}
        img_file = coco.loadImgs(int(idx))[0]['img_file']
        obj_pose = np.loadtxt(anno['pose_file'])
        intrin = np.loadtxt(anno['intrinsic_file'])
        cam_pose = data_utils.obj2cam(obj_pose)
        obj_pose_quat = data_utils.mtx2quat(obj_pose)
        with open(anno['mask_file'], 'r') as f:
            mask = json.load(f)
        if load_in_ram:
            mask = data_utils.load_mask(mask, res=res)
            img, img_norm, scaling = data_utils.load_rgb(img_file, res=res)
            img = img * mask[..., np.newaxis]
            img_norm = img_norm * mask[..., np.newaxis]
            catagory_data[catagory_name]['data'][img_id] = {'img': img,
                                                            'img_norm': img_norm,
                                                            'scaling': scaling,
                                                            'intrinsic': intrin,
                                                            'obj_pose': obj_pose,
                                                            'obj_pose_quat': obj_pose_quat,
                                                            'cam_pose': cam_pose,
                                                            'mask': mask,
                                                            'catagory': catagory_name}
        else:
            catagory_data[catagory_name]['data'][img_id] = {'img_file': img_file,
                                                            'intrinsic': intrin,
                                                            'obj_pose': obj_pose,
                                                            'obj_pose_quat': obj_pose_quat,
                                                            'cam_pose': cam_pose,
                                                            'mask': mask,
                                                            'catagory': catagory_name}
        data_idx[idx-1] = {'catagory': catagory_name,
                           'img_id': img_id}
        img_id += 1

    idx_start = 0
    for catagory_name in catagory_data.keys():
        catagory_data[catagory_name]['index_range'] = [idx_start, idx_start + len(catagory_data[catagory_name]['data'])]
        idx_start += len(catagory_data[catagory_name]['data'])

    init_idx = list(data_idx.keys())[0]
    if init_idx != 0:
        data_idx_update = {}
        for i in data_idx.keys():
            data_idx_update[i-init_idx] = data_idx[i]
        data_idx = data_idx_update
    return catagory_data, data_idx


def add_view_density(data):
    for catagory_name in data.keys():
        all_cam_poses = [data[catagory_name]['data'][i]['cam_pose'] for i in range(len(data[catagory_name]['data']))]
        all_cam_poses = np.stack(all_cam_poses)[:, :3, :3]
        mean_density, ind_dict = data_utils.sampled_view_density(all_cam_poses)
        for i in range(len(data[catagory_name]['data'])):
            data[catagory_name]['data'][i]['density'] = ind_dict[i]
        data[catagory_name]['avg_density'] = mean_density
    return data


class NeRFDataset(Dataset):
    def __init__(
            self,
            anno_file,
            num_leaf,
            split,
            pad=True,
            shape2d=1000,
            shape3d=2000,
            pad_val=0,
            load_pose_gt=False,
            num_source_views=10,
            num_total_views=30,
            resolution=128,
            load_in_ram=False,
            dataset_dirs=None
    ):
        super(Dataset, self).__init__()

        self.coco = COCO(anno_file)
        self.anns = np.array(self.coco.getImgIds())
        self.num_leaf = num_leaf

        self.split = split
        self.pad = pad
        self.shape2d = shape2d
        self.shape3d = shape3d
        self.pad_val = pad_val
        self.load_pose_gt = load_pose_gt
        self.num_source_views = num_source_views
        self.num_total_views = num_source_views
        self.resolution = resolution
        self.imgnet_norm = True
        self.load_in_ram = load_in_ram
        self.data, self.data_idx = load_data(self.anns, self.coco, self.resolution, load_in_ram)
        self.data = add_view_density(self.data)
        self.dataset_dirs = dataset_dirs

    def read_anno_multiview(self, anno, catagory_anno, render_id, radius, avg_density):

        # Load camera poses
        multiview_poses = data_utils.load_multiview_cam_poses(catagory_anno)
        render_obj_pose = anno['obj_pose']
        render_cam_pose = anno['cam_pose']
        render_obj_pose_quat = anno['obj_pose_quat']
        src_views = self.num_source_views
        feat_id_pool = data_utils.get_nearest_pose_ids(render_cam_pose, multiview_poses, src_views, tar_id=render_id, angular_dist_method='matrix')
        feat_id = np.random.choice(feat_id_pool, self.num_source_views, replace=False)
        assert render_id not in feat_id
        src_poses = multiview_poses[feat_id]

        # Load images and masks
        if self.load_in_ram:
            render_mask = anno['mask']
            render_rgb = anno['img']
            render_rgb_norm = anno['img_norm']
            scaling = anno['scaling']
            src_rgbs = np.stack([catagory_anno[feat_id[i]]['img_norm'] for i in range(src_views)])
        else:
            rgb_file = anno['img_file']
            mask_file = catagory_anno[render_id]['mask']
            render_mask = data_utils.load_mask(mask_file, res=self.resolution)
            # vis_mask(render_mask)
            render_rgb, render_rgb_norm, scaling = data_utils.load_rgb(rgb_file, res=self.resolution)
            src_rgbs = data_utils.load_src_rgbs(catagory_anno, ids=feat_id, res=self.resolution)

        # Load intrinsics
        render_intrin, src_intrin = data_utils.load_intrinsics(catagory_anno, render_id, feat_id, scaling)

        # Load depth range
        depth_range = data_utils.get_depth_range(render_obj_pose, radius)

        return {'rgb': render_rgb,
                'rgb_norm': render_rgb_norm,
                'mask': render_mask,
                'render_cam': render_cam_pose,
                'render_obj': render_obj_pose_quat,
                'src_rgbs': src_rgbs,
                'src_cam': src_poses,
                'render_intrin': render_intrin,
                'src_intrin': src_intrin,
                'depth_range': depth_range,
                # 'density_offset': (1/anno['density']) ** 2}
                'density_offset': (avg_density / anno['density'])}


    def read_anno(self, index):
        catagory = self.data_idx[index]['catagory']
        img_id = self.data_idx[index]['img_id']
        anno = self.data[catagory]['data'][img_id]
        catagory_annos = self.data[catagory]['data']
        radius = self.data[catagory]['radius']
        avg_density = self.data[catagory]['avg_density']
        data = self.read_anno_multiview(anno, catagory_annos, img_id, radius, avg_density)
        return data#, conf_matrix

    def __getitem__(self, index):
        return self.read_anno(index)

    def __len__(self):
        return len(self.data_idx)