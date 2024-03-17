import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
import open3d as o3d
from pytorch3d.ops import sample_farthest_points
from torch.nn.utils.rnn import pad_sequence
from src.utils import pose_utils


def sample_voxel(x3d, x3d_norm, lengthmap, inv_crd, img_crd, num_voxels, num_crd, training=True):
    pad_val = -1

    # farthest point sample voxels to obtain a good coverage of the object
    x3d_sample, fps_idx_voxel = sample_farthest_points(x3d, lengths=lengthmap, K=num_voxels*2, random_start_point=True)
    if training:

        # additionally sample voxels that are visible by the query image
        fps_idx_voxel_crd = pose_utils.sample_from_img_crd(img_crd, num=num_crd)
        fps_idx_voxel = torch.cat((fps_idx_voxel, fps_idx_voxel_crd), dim=1)
    # fps_idx_voxel = torch.unique(fps_idx_voxel, return_counts=False)

    # update the according to the sampled index
    inv_crd_pad = []
    for i, crd in enumerate(inv_crd):
        inv_crd_pad += [crd[f.item()] for f in fps_idx_voxel[i]]
    x3d_sample = torch.gather(x3d, 1, fps_idx_voxel.unsqueeze(-1).repeat(1, 1, 3))
    x3d_norm = torch.gather(x3d_norm, 1, fps_idx_voxel.unsqueeze(-1).repeat(1, 1, 3))
    inv_crd = pad_sequence(inv_crd_pad, batch_first=True, padding_value=pad_val)
    return fps_idx_voxel, inv_crd, x3d_sample, x3d_norm


def sample_img(inv_crd, trans_all, rgb_all, intr_all, extr_all, num_imgs, num_voxels, num_crd, training=True):
    lengthmap = trans_all[1]
    trans_all = trans_all[0]
    temp_sampling = num_voxels * 2 + num_crd if training else num_voxels * 2#inv_crd.shape[0]##

    # farthest point sample reference images using their camera poses' translation part
    _, fps_idx_img = sample_farthest_points(trans_all, lengths=lengthmap, K=num_imgs, random_start_point=True)
    rgb_all = torch.stack([rgb_all[i][fps_idx_img[i]] for i in range(len(rgb_all))])
    intr_all = torch.stack([intr_all[i][fps_idx_img[i]] for i in range(len(intr_all))])
    extr_all = torch.stack([extr_all[i][fps_idx_img[i]] for i in range(len(extr_all))])
    fps_idx_img = fps_idx_img.to(torch.int16).repeat_interleave(temp_sampling, dim=0).unsqueeze(1)
    # if training:
    #     _, fps_idx_img = sample_farthest_points(trans_all, lengths=lengthmap, K=num_imgs, random_start_point=True)
    #     rgb_all = torch.stack([rgb_all[i][fps_idx_img[i]] for i in range(len(rgb_all))])
    #     intr_all = torch.stack([intr_all[i][fps_idx_img[i]] for i in range(len(intr_all))])
    #     extr_all = torch.stack([extr_all[i][fps_idx_img[i]] for i in range(len(extr_all))])
    #     fps_idx_img = fps_idx_img.to(torch.int16).repeat_interleave(temp_sampling, dim=0).unsqueeze(1)
    # else:
    #     fps_idx_img = torch.arange(trans_all.shape[1]).to(torch.int16).unsqueeze(0).repeat_interleave(temp_sampling, dim=0).unsqueeze(1)
    #     rgb_all = rgb_all[0].unsqueeze(0)
    #     intr_all = intr_all[0].unsqueeze(0)
    #     extr_all = extr_all[0].unsqueeze(0)

    # main idea of all the following code is to know which voxel is visible by which image since everything is sampled
    img_crd_padded = inv_crd[..., 0].to(torch.int16).unsqueeze(-1)
    t_crd = inv_crd[..., 1]
    all_match = (img_crd_padded == fps_idx_img)
    all_idx = all_match.nonzero(as_tuple=True)

    num_matches_per_voxel = all_match.any(dim=-1).sum(dim=1)
    match_idx = pose_utils.sample_from_all_matches(num_matches_per_voxel, num_voxels, temp_sampling)

    num_matches_per_voxel_sampled = torch.gather(num_matches_per_voxel.view(-1, temp_sampling), 1, match_idx)
    assert num_matches_per_voxel_sampled.min() > 0
    inv_crd_idx = all_idx[0], all_idx[1]

    img_crd_flat = all_idx[2]#[match_idx]
    t_crd_flat = t_crd[inv_crd_idx]
    inv_crd_flat = torch.stack((img_crd_flat, t_crd_flat), dim=-1)
    inv_crd = list(torch.split(inv_crd_flat, num_matches_per_voxel.tolist()))
    index_rectification = (torch.arange(match_idx.shape[0]) * temp_sampling).repeat(match_idx.shape[1], 1).t()
    match_idx_flat = (match_idx + index_rectification).view(-1, 1)

    visibility = list(torch.split(torch.ones_like(img_crd_flat), num_matches_per_voxel.tolist()))
    inv_crd = pad_sequence(inv_crd, batch_first=True, padding_value=0)#.view(rgb_all.shape[0], num_voxels, -1, 2).to(device)
    inv_crd = inv_crd[match_idx_flat.view(-1)]
    inv_crd = inv_crd.view(rgb_all.shape[0], num_voxels, -1, 2)
    visibility = pad_sequence(visibility, batch_first=True, padding_value=0)
    visibility = visibility[match_idx_flat.view(-1)]
    visibility = visibility.view(rgb_all.shape[0], num_voxels, -1)
    if visibility.reshape(-1, 50).sum(dim=-1).min() == 0:
        print('visibility error')
    return inv_crd, rgb_all, intr_all, extr_all, visibility, match_idx


class DataSampler():
    def __init__(self, batch, voxel_samples, image_samples, crd_samples, sigma):
        self.batch = batch
        self.voxel_samples = voxel_samples
        self.image_samples = image_samples
        self.crd_samples = crd_samples
        self.voxel_idx = None
        self.device = None
        self.batch_size = batch['rgb'].shape[0]
        self.sigma = sigma
        self.gt_sim = None
        self.training = False

    def sample_data(self):
        self._sample_data()
        if self.training:
            render_pixel, lengthmap = self.get_gt_sim()
            self.sample_ray(render_pixel, lengthmap)
        self.move_to_device()
        return self.batch

    def update_status(self, device, training=True):
        self.device = device
        self.training = training

    def _sample_data(self):
        lengthmap = self.batch['voxel'][1]
        x3d = self.batch['voxel'][0]
        x3d_norm = self.batch['voxel_norm']
        inv_crd = self.batch['inv_crd']
        ref_rgbs = self.batch['ref_rgbs']
        ref_intrin = self.batch['ref_intr']
        ref_pose = self.batch['ref_pose']
        ref_trans = self.batch['ref_trans']
        img_crd = self.batch['crd'] if self.training else None

        # Sample voxel from all voxels of the object
        voxel_idx, inv_crd, x3d, x3d_norm = sample_voxel(x3d,
                                                         x3d_norm,
                                                         lengthmap,
                                                         inv_crd,
                                                         img_crd,
                                                         self.voxel_samples,
                                                         self.crd_samples,
                                                         self.training)

        # Sample reference images from all reference images of the object
        inv_crd, ref_rgbs, ref_intr, ref_pose, visibility, voxel_idx_sampled = sample_img(inv_crd,
                                                                                          ref_trans,
                                                                                          ref_rgbs,
                                                                                          ref_intrin,
                                                                                          ref_pose,
                                                                                          self.image_samples,
                                                                                          self.voxel_samples,
                                                                                          self.crd_samples,
                                                                                          self.training)
        self.voxel_idx = torch.gather(voxel_idx, 1, voxel_idx_sampled).to(torch.int16)
        self.batch['inv_crd'] = inv_crd
        self.batch['voxel'] = torch.gather(x3d, 1, voxel_idx_sampled.unsqueeze(-1).repeat(1, 1, 3))
        self.batch['voxel_norm'] = torch.gather(x3d_norm, 1, voxel_idx_sampled.unsqueeze(-1).repeat(1, 1, 3))
        self.batch['ref_rgbs'] = ref_rgbs
        self.batch['ref_intr'] = ref_intr
        self.batch['ref_pose'] = ref_pose
        self.batch['visibility'] = visibility

    def get_gt_sim(self):
        img_crd = self.batch['crd'][0]
        img_crd_pos = img_crd[..., :3]
        img_crd_t = img_crd[..., -2]
        img_crd_idx = img_crd[..., -1].to(torch.int16)
        pixel = self.batch['pixel'].clone()
        intrinsic = self.batch['query_intr']
        extrinsic = self.batch['query_pose']

        # idx[i][0]: index in img_crd voxel, idx[i][1]: index in sample_idx
        matches = (img_crd_idx.unsqueeze(-1) == self.voxel_idx.unsqueeze(1))
        idx = matches.nonzero(as_tuple=True)
        idx_crd = idx[0], idx[1]
        idx_sample = idx[0], idx[2]

        indices, lengthmap = torch.unique_consecutive(idx[0], return_counts=True)
        img_crd_t = pad_sequence(list(torch.split(img_crd_t[idx_crd], lengthmap.tolist())), batch_first=True,
                                 padding_value=0)
        x3d_sample = pad_sequence(list(torch.split(img_crd_pos[idx_crd], lengthmap.tolist())), batch_first=True,
                                  padding_value=0).to(torch.float32)
        if indices.shape[0] != extrinsic.shape[0]:
            total_idx = torch.arange(extrinsic.shape[0])
            total_samples = torch.zeros((total_idx.shape[0], x3d_sample.shape[1], 3))
            total_samples[indices] = x3d_sample
            x3d_sample = total_samples

        # Project sampled voxels' coordinates to query image and generate the ground truth similarity matrix
        x2d_proj = pose_utils.batch_projection_2d(x3d_sample, intrinsic, extrinsic)  # y, x
        batch_weights_2d = pose_utils.reproj_gaussian_weight(pixel, x2d_proj, img_crd_t, self.sigma)
        gt_sim = pose_utils.fill_sim_mtx(batch_weights_2d, idx_sample, self.voxel_samples, lengthmap)

        self.gt_sim = gt_sim
        return x2d_proj, lengthmap

    def sample_ray(self, render_pixel, lengthmap):
        pass
        # intrinsics = self.batch['query_intr']
        # extrinsics = self.batch['query_pose']
        # rays_d = (extrinsics[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(render_pixel)).transpose(1, 2)
        # # rays_d = rays_d.reshape(-1, 3)
        # rays_o = extrinsics[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)

    def move_to_device(self):
        self.batch['rgb'] = self.batch['rgb'].to(self.device, non_blocking=True)
        self.batch['pixel'] = self.batch['pixel'].to(self.device, non_blocking=True)
        self.batch['rgb_norm'] = self.batch['rgb_norm'].to(self.device, non_blocking=True)
        self.batch['ref_rgbs'] = self.batch['ref_rgbs'].to(self.device, non_blocking=True)
        self.batch['query_pose'] = self.batch['query_pose'].to(self.device, non_blocking=True)
        self.batch['query_pose_quat'] = self.batch['query_pose_quat'].to(self.device, non_blocking=True)
        self.batch['ref_pose'] = self.batch['ref_pose'].to(self.device, non_blocking=True)
        self.batch['query_intr'] = self.batch['query_intr'].to(self.device, non_blocking=True)
        self.batch['ref_intr'] = self.batch['ref_intr'].to(self.device, non_blocking=True)
        self.batch['voxel'] = self.batch['voxel'].to(self.device, non_blocking=True)
        self.batch['voxel_norm'] = self.batch['voxel_norm'].to(self.device, non_blocking=True)
        self.batch['inv_crd'] = self.batch['inv_crd'].to(self.device, non_blocking=True)
        self.batch['visibility'] = self.batch['visibility'].to(self.device, non_blocking=True)
        if self.training:
            self.batch['gt_sim'] = self.gt_sim.to(self.device, non_blocking=True)