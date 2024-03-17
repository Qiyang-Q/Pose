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
from src.utils import data_utils, pose_utils
import imageio
import os.path as osp
import matplotlib.pyplot as plt


def load_data(anns, coco, res=128, sample_res=64, load_in_ram=False):
    data_idx = {}
    catagory_data = {}
    for idx in anns:
        ann_id = coco.getAnnIds(imgIds=idx)
        anno = coco.loadAnns(ann_id)[0]
        catagory_name = anno['pose_file'].split('/')[-4]
        if catagory_name not in catagory_data:
            img_id = 0
            catagory_path = '/'.join(anno['pose_file'].split('/')[:-3])
            bbox3d_file = os.path.join(catagory_path, 'box3d_corners.txt')
            radius = data_utils.get_radius(bbox3d_file)
            voxel_path = os.path.join(catagory_path, 'voxel_position.npy')
            crd_path = os.path.join(catagory_path, 'correspondence.npy')
            voxel = torch.from_numpy(np.load(voxel_path)).to(torch.float32)
            voxel_norm = data_utils.norm_coor_3d(voxel)
            crd = np.load(crd_path, allow_pickle=True).item()
            inv_crd = data_utils.get_inverse_crd(crd)
            voxel, voxel_norm, inv_crd, crd = data_utils.sample_crd(voxel, voxel_norm, inv_crd, crd, crd_threshold=20)
            crd = {k: torch.from_numpy(v).to(torch.float32) for k, v in crd.items()}
            catagory_data[catagory_name] = {'radius': radius,
                                            'data': {},
                                            'voxel': voxel,
                                            'voxel_norm': voxel_norm,
                                            'crd': crd,
                                            'inv_crd': inv_crd,
                                            }
        img_file = coco.loadImgs(int(idx))[0]['img_file']
        obj_pose = np.loadtxt(anno['pose_file'])
        obj_pose_quat = data_utils.mtx2quat(obj_pose)
        intrin = np.loadtxt(anno['intrinsic_file'])
        cam_pose = data_utils.obj2cam(obj_pose)

        ##### debug #####
        mask_dir = '/' + os.path.join(*img_file.split('/')[:-2], 'mask')
        mask_file = os.path.join(mask_dir, str(img_id) + '.json')
        seq_id = int(mask_dir[-6])
        if seq_id == 4:
            masked_id = os.listdir(mask_dir)
            masked_id = [int(i.split('.')[0]) for i in masked_id]
            if img_id not in masked_id:
                img_id += 1
                continue
            else:
                with open(mask_file, 'r') as f:
                    mask = json.load(f)
        # if seq_id == 4:
        #     mask = None
        # else:
        else:
            with open(anno['mask_file'], 'r') as f:
                mask = json.load(f)
            mask = data_utils.load_mask(mask, res=res)
        if load_in_ram:
            img, img_norm, scaling = data_utils.load_rgb(img_file, res=res)
            coord = data_utils.get_coords(sample_res)
            catagory_data[catagory_name]['data'][img_id] = {'img': img,
                                                            'img_norm': img_norm,
                                                            'scaling': scaling,
                                                            'intrinsic': intrin,
                                                            'obj_pose': obj_pose[:3, :],
                                                            'obj_pose_quat': obj_pose_quat,
                                                            'cam_pose': cam_pose[:3, :],
                                                            'mask': mask,
                                                            'catagory': catagory_name,
                                                            'coord': coord}
        else:
            if seq_id == 4:
                mask = data_utils.load_mask(mask, res=res)
            catagory_data[catagory_name]['data'][img_id] = {'img_file': img_file,
                                                            'intrinsic': intrin,
                                                            'obj_pose': obj_pose[:3, :],
                                                            'obj_pose_quat': obj_pose_quat,
                                                            'cam_pose': cam_pose[:3, :],
                                                            'mask': mask,
                                                            'catagory': catagory_name}
        data_idx[idx-1] = {'catagory': catagory_name,
                           'img_id': img_id}

        img_id += 1

    idx_start = 0
    for catagory_name in catagory_data.keys():
        catagory_data[catagory_name]['index_range'] = [idx_start, idx_start + len(catagory_data[catagory_name]['data'])]
        idx_start += len(catagory_data[catagory_name]['data'])

    data_idx_update = {}
    for i, key in enumerate(data_idx.keys()):
        data_idx_update[i] = data_idx[key]
    data_idx = data_idx_update
    return catagory_data, data_idx


def add_view_density(data):
    for catagory_name in data.keys():
        all_cam_poses = [data[catagory_name]['data'][i]['cam_pose'] for i in range(len(data[catagory_name]['data']))]
        all_cam_poses = np.stack(all_cam_poses)[:, :3, :3]
        mean_density, ind_dict = data_utils.sampled_view_density(all_cam_poses)
        all_density = []
        for i in range(len(data[catagory_name]['data'])):
            # cam_pose = data[catagory_name]['data'][i]['cam_pose'][:3, :3]
            # density = data_utils.get_view_density(cam_pose, all_cam_poses)
            # all_density.append(density)
            data[catagory_name]['data'][i]['density'] = ind_dict[i]
        # mean_density = np.mean(np.stack(all_density))
        data[catagory_name]['avg_density'] = mean_density
    return data


class PoseDataset(Dataset):
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
            sample_res=64,
            load_in_ram=False,
            dataset_dirs=None,
            query_dir=None,
            test_anno=None
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
        self.sample_res = sample_res
        self.imgnet_norm = True
        self.load_in_ram = load_in_ram
        self.data, self.data_idx = load_data(self.anns, self.coco, self.resolution, self.sample_res, self.load_in_ram)
        # self.data = add_view_density(self.data)
        self.dataset_dirs = dataset_dirs
        if self.split == 'test':
            self.test_coco = COCO(test_anno)
            self.test_anns = np.array(self.test_coco.getImgIds())
            self.test_dir = query_dir
            # self.test_data, self.test_data_idx = self.load_test_data()
            self.test_data, self.test_data_idx = load_data(self.test_anns, self.test_coco, self.resolution, self.sample_res)

    def load_test_data(self):
        test_data = {}
        test_data_idx = {}
        idx = 0
        for dir in self.test_dir:
            sequence = dir.split(' ')[-1]
            catagory_dir = os.path.join(dir.split(' ')[0], sequence)
            catagory = catagory_dir.split('/')[-2]
            rgb_dir = os.path.join(catagory_dir, 'color')
            intr_dir = os.path.join(catagory_dir, 'intrin_ba')
            extr_dir = os.path.join(catagory_dir, 'poses_ba')
            test_data[catagory] = {}
            for i in range(len(os.listdir(rgb_dir))):
                rgb_file = os.path.join(rgb_dir, f'{i}.png')
                intr_file = os.path.join(intr_dir, f'{i}.txt')
                pose_file = os.path.join(extr_dir, f'{i}.txt')
                obj_pose = np.loadtxt(pose_file)
                cam_pose = data_utils.obj2cam(obj_pose)
                pose_quat = data_utils.mtx2quat(obj_pose)
                intr = np.loadtxt(intr_file)
                test_data[catagory][i] = {'img_file': rgb_file,
                                          'intrinsic': intr,
                                          'obj_pose': obj_pose,
                                          'cam_pose': cam_pose,
                                          'obj_pose_quat': pose_quat}
                test_data_idx[idx] = {'catagory': catagory,
                                      'img_id': i}
                idx += 1
        return test_data, test_data_idx

    def read_anno_multiview(self, anno, catagory_anno, query_id, catagory_info, catagory_name):

        # Load camera poses
        ref_poses = data_utils.load_multiview_obj_poses(catagory_anno)
        query_obj_pose = anno['obj_pose']
        query_obj_pose_quat = anno['obj_pose_quat']
        query_intr = anno['intrinsic']
        ref_ids = np.stack([i for i in catagory_anno.keys()])
        query_cam_pose = anno['cam_pose']
        res_scaling = self.sample_res / self.resolution

        # Load images and masks
        if self.load_in_ram and self.split != 'test':
            mask = anno['mask']
            query_rgb = anno['img']
            query_rgb_norm = anno['img_norm']
            scaling = anno['scaling']
            coord = anno['coord']
            ref_rgbs = np.stack([catagory_anno[i]['img_norm'] for i in catagory_anno.keys()])
            ref_masks = np.stack([catagory_anno[i]['mask'] for i in catagory_anno.keys()])
        else:
            mask = anno['mask']
            rgb_file = anno['img_file']
            query_rgb, query_rgb_norm, scaling = data_utils.load_rgb(rgb_file, res=self.resolution)
            coord = data_utils.get_coords(self.sample_res)
            if self.split == 'test':
                ref_rgbs = np.stack([catagory_anno[i]['img_norm'] for i in catagory_anno.keys()])
                ref_masks = np.stack([catagory_anno[i]['mask'] for i in catagory_anno.keys()])
            else:
                ref_rgbs = data_utils.load_src_rgbs(catagory_anno, ids=ref_ids, res=self.resolution)

        # vis_src_views(query_rgb, ref_rgbs)
        # data_utils.vis(render_rgb * render_mask[..., None])
        query_rgb_norm = query_rgb_norm * mask[..., np.newaxis]
        ref_rgbs = ref_rgbs * ref_masks[..., np.newaxis]
        # vis_src_views(query_rgb_norm, ref_rgbs)
        # Load intrinsics
        query_intr = torch.from_numpy(query_intr).to(torch.float32)
        query_intrin = data_utils.rectify_intrinsic(query_intr, scaling*res_scaling)[:3, :3]
        ref_intrin = [catagory_anno[id]['intrinsic'] for id in ref_ids]
        ref_intrin = torch.from_numpy(np.array(ref_intrin)).to(torch.float32)
        ref_intrin = data_utils.rectify_intrinsic(ref_intrin, scaling)[:, :3, :3]

        #
        # # Load depth range
        radius = catagory_info['radius']
        depth_range = data_utils.get_depth_range(query_obj_pose, radius)

        return {'rgb': torch.from_numpy(query_rgb).to(torch.float32),
                'rgb_norm': torch.from_numpy(query_rgb_norm).to(torch.float32),
                # 'mask': query_mask,
                'query_cam': torch.from_numpy(query_cam_pose).to(torch.float32),
                'query_pose': torch.from_numpy(query_obj_pose).to(torch.float32),
                'query_pose_quat': torch.from_numpy(query_obj_pose_quat).to(torch.float32),
                'ref_rgbs': torch.from_numpy(ref_rgbs).to(torch.float32),
                'ref_pose': torch.from_numpy(ref_poses).to(torch.float32),
                'query_intr': query_intrin,
                'ref_intr': ref_intrin,
                'pixel': coord,
                'voxel': catagory_info['voxel'],
                'voxel_norm': catagory_info['voxel_norm'],
                'crd': catagory_info['crd'][query_id] if self.split != 'test' else None,
                'inv_crd': catagory_info['inv_crd'],
                'query_id': query_id,
                'name': catagory_name,
                'depth_range': depth_range.to(torch.float32),
                'ref_ids': ref_ids}


    def read_anno(self, index):
        """
        Read image, 2d info and 3d info.
        Pad 2d info and 3d info to a constant size.
        """
        if self.split == 'test':
            catagory = self.test_data_idx[index]['catagory']
            img_id = self.test_data_idx[index]['img_id']
            # anno = self.test_data[catagory][img_id]
            anno = self.test_data[catagory]['data'][img_id]
        else:
            catagory = self.data_idx[index]['catagory']
            img_id = self.data_idx[index]['img_id']
            anno = self.data[catagory]['data'][img_id]
        catagory_annos = self.data[catagory]['data']
        catagory_info = self.data[catagory]
        data = self.read_anno_multiview(anno, catagory_annos, img_id, catagory_info, catagory)
        return data#, conf_matrix

    def __getitem__(self, index):
        return self.read_anno(index)

    def __len__(self):
        if self.split == 'test':
            return len(self.test_data_idx)
        else:
            return len(self.data_idx)