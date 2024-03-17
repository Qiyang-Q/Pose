import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import numpy as np
import open3d as o3d
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points
import torch.nn.functional as F
import os
from torch.nn.utils.rnn import pad_sequence, pack_sequence
# plt.ioff()


def project_3d_to_2d_batch(P, intrinsics, extrinsics):
    # Convert P to homogeneous coordinates
    device = P.device
    P_homo = torch.cat([P, torch.tensor([1.0]).to(device)])
    P_homo = P_homo.repeat(len(intrinsics), 1).unsqueeze(2)  # Shape: [N, 4, 1]

    # Combine intrinsic and extrinsic for batch matrix multiplication
    combined = torch.bmm(intrinsics, extrinsics)

    # Project using the combined matrices
    p_prime = torch.bmm(combined, P_homo)

    # Convert back to non-homogeneous 2D coordinates
    p = torch.stack([p_prime[:, 0, 0] / p_prime[:, 2, 0], p_prime[:, 1, 0] / p_prime[:, 2, 0]], dim=1)
    p = p.flip(-1)
    return p

# @torch.jit.script
def multiview_back_projection(P, I, E):
    batch_size = P.shape[0]
    num_point = P.shape[1]

    # Augment the 3D points and compute their projections using the extrinsic matrices
    P_homo = torch.cat([P, torch.ones(batch_size, num_point, 1, device=P.device)], dim=-1).unsqueeze(1)  # [B, 1, N, 4]
    projected = (E @ P_homo.transpose(-1, -2)).transpose(-1, -2)  # [B, M, N, 3]

    # Project using the intrinsic matrices and normalize to obtain 2D coordinates
    x2d_proj = (I @ projected.transpose(-1, -2)).permute(0, 3, 1, 2)  # [B, N, M, 3]
    x2d_proj = x2d_proj[..., :2] / x2d_proj[..., 2:3]  # [B, N, M, 2]

    return x2d_proj.flip(-1)


def combine_img(img1, img2):
    if isinstance(img1, torch.Tensor):
        img1 = img1.numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.numpy()
    img = np.concatenate([img1, img2], axis=1)
    return img


def quat2mat(quat):
    bs = quat.shape[0]
    T_vectors, R_quats = quat.split([3, 4], dim=-1)
    R_matrix = R.from_quat(R_quats[:, [1, 2, 3, 0]].cpu().numpy()).as_matrix()  # (n, 3, 3)
    mat = np.concatenate([R_matrix, T_vectors.reshape(bs, 3, 1).cpu().numpy()], axis=-1)
    return mat


def projection_2d(x3d, intrinsic, extrinsic):
    N = x3d.size(0)
    ones = torch.ones(N, 1, device=x3d.device, dtype=x3d.dtype)
    x3d_homogeneous = torch.cat([x3d, ones], dim=1)
    # Multiply with extrinsic
    points_cam = torch.mm(x3d_homogeneous, extrinsic.t())
    # Multiply with intrinsic
    x2d_homogeneous = torch.mm(points_cam, intrinsic.t())
    # Convert to cartesian coordinates
    x2d_proj = x2d_homogeneous[:, :2] / x2d_homogeneous[:, 2].unsqueeze(-1)
    x2d_proj = x2d_proj.flip(-1)
    return x2d_proj#, x2d_homogeneous[:, 2]


def batch_projection_2d(x3d, intrinsic, extrinsic):
    B, N, _ = x3d.size()

    ones = torch.ones(B, N, 1, device=x3d.device, dtype=x3d.dtype)
    x3d_homogeneous = torch.cat([x3d, ones], dim=2)

    # Multiply with extrinsic
    points_cam = torch.bmm(x3d_homogeneous, extrinsic.transpose(1, 2))

    # Multiply with intrinsic
    x2d_homogeneous = torch.bmm(points_cam, intrinsic.transpose(1, 2))

    # Convert to cartesian coordinates
    x2d_proj = x2d_homogeneous[:, :, :2] / x2d_homogeneous[:, :, 2].unsqueeze(-1)

    # Flip the coordinate to y, x for alignment
    x2d_proj = x2d_proj.flip(-1)

    return x2d_proj


def norm_coor_2d(coor, res):
    resize_factor = torch.tensor([res-1., res-1.]).to(coor.device)[None, None, :]
    coor = coor / resize_factor
    return coor


def normalize_coord(x2d_proj, res=64):
    x2d_proj[:, 0] = (x2d_proj[:, 0] - x2d_proj[:, 0].min()) / (x2d_proj[:, 0].max() - x2d_proj[:, 0].min()) * (res - 1)
    x2d_proj[:, 1] = (x2d_proj[:, 1] - x2d_proj[:, 1].min()) / (x2d_proj[:, 1].max() - x2d_proj[:, 1].min()) * (res - 1)
    return x2d_proj


def normalize(pixel_locations, res):
    resize_factor = torch.tensor([res-1., res-1.]).to(pixel_locations.device)[None, None, :]
    normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.
    return normalized_pixel_locations.flip(-1)


def norm_coor_3d(coor):
    coor_x, coor_y, coor_z = coor[..., 0], coor[..., 1], coor[..., 2]
    min_x, max_x = torch.min(coor_x, dim=-1, keepdim=True)[0], torch.max(coor_x, dim=-1, keepdim=True)[0]
    min_y, max_y = torch.min(coor_y, dim=-1, keepdim=True)[0], torch.max(coor_y, dim=-1, keepdim=True)[0]
    min_z, max_z = torch.min(coor_z, dim=-1, keepdim=True)[0], torch.max(coor_z, dim=-1, keepdim=True)[0]
    coor_x = (coor_x - min_x) / (max_x - min_x)
    coor_y = (coor_y - min_y) / (max_y - min_y)
    coor_z = (coor_z - min_z) / (max_z - min_z)
    return torch.stack((coor_x, coor_y, coor_z), dim=-1)


def parse_similarity(similarity, src_pos, tgt_pos, weightmap):
    # similarity = similarity[..., :-1]

    # weightmap is a 2D map from query feature map
    # indicating which pixels are more likely to have higher weights for correspondence
    weight = weightmap.view(weightmap.shape[0], -1, weightmap.shape[-1])
    weight = weight.unsqueeze(1).repeat(1, similarity.shape[1], 1, 1)

    # Align similarity and weightmap
    similarity_reshape = similarity.unsqueeze(-1).repeat(1, 1, 1, weightmap.shape[-1])

    # Softmax over the similarity matrix in the row direction if not using dual softmax
    # w2d = dual_softmax(similarity_reshape + weight)
    w2d = F.softmax(similarity_reshape + weight, dim=-2)
    similarity = F.softmax(similarity, dim=-1)
    # similarity = dual_softmax(similarity)

    # flatten similarity matrix, extend 2D coordinates and 3D coordinates to obtain all correspondences
    confidence = similarity.reshape(similarity.shape[0], -1, 1)
    w2d = w2d.reshape(similarity.shape[0], -1, weightmap.shape[-1])
    x3d_aug = tgt_pos[:, :, None, :].repeat(1, 1, similarity.shape[2], 1).reshape(similarity.shape[0], -1, 3)
    x2d_aug = src_pos[:, None, :, :].repeat(1, similarity.shape[1], 1, 1).reshape(similarity.shape[0], -1, 2)

    return confidence, x3d_aug, x2d_aug, w2d

def hard_threshold(confidence, x3d_aug, x2d_aug, w2d_aug, init_sample, threshold):
    rectified_conf = w2d_aug.mean(dim=-1).unsqueeze(-1)
    selected = rectified_conf >= threshold
    num_selected = selected.sum(dim=1).max().item()
    num_selected = max(num_selected, init_sample)
    _, selected_idx = torch.topk(rectified_conf, num_selected, dim=1)
    x3d_select = x3d_aug.gather(1, selected_idx.repeat(1, 1, 3))
    x2d_select = x2d_aug.gather(1, selected_idx.repeat(1, 1, 2))
    w2d_select = w2d_aug.gather(1, selected_idx.repeat(1, 1, w2d_aug.shape[-1]))
    conf_select = confidence.gather(1, selected_idx)
    return conf_select, x3d_select, x2d_select, w2d_select

def prob_sampling(confidence, x3d_aug, x2d_aug, w2d_aug, init_sample):
    rectified_conf = w2d_aug.mean(dim=-1).unsqueeze(-1)# * confidence

    # Sample correspondences based on the probability of the weights
    selected_idx = torch.multinomial(rectified_conf.squeeze(-1), init_sample, replacement=False).unsqueeze(-1)

    x3d_select = x3d_aug.gather(1, selected_idx.repeat(1, 1, 3))
    x2d_select = x2d_aug.gather(1, selected_idx.repeat(1, 1, 2))
    w2d_select = w2d_aug.gather(1, selected_idx.repeat(1, 1, w2d_aug.shape[-1]))
    conf_select = confidence.gather(1, selected_idx)
    return conf_select, x3d_select, x2d_select, w2d_select

def sample_similarity(confidence, x3d_aug, x2d_aug, w2d_aug, threshold=0.1, strategy='hard', init_sample=500):
    if strategy == 'hard':
        return hard_threshold(confidence, x3d_aug, x2d_aug, w2d_aug, init_sample, threshold)
    if strategy == 'prob':
        return prob_sampling(confidence, x3d_aug, x2d_aug, w2d_aug, init_sample)


def reproj_gaussian_weight(P, C, T, sigma):
    """
    P: tensor of shape [B, N, 2]
    C: tensor of shape [B, M, 2]
    sigma: float

    Returns: tensor of shape [B, N, M], where each row corresponds to a point in P
             and each column corresponds to a center in C.
    """
    # Expand dimensions of P and C to make them [N, 1, 2] and [1, M, 2]
    P = P.unsqueeze(1)
    C = C.unsqueeze(2)

    # Broadcasting the subtraction operation to compute pairwise differences
    # This results in a tensor D of shape [N, M, 2]
    D = P - C

    # Compute squared Euclidean distances; results in [N, M]
    squared_distances = (D ** 2).sum(-1)

    # Compute weights
    weights = torch.exp(-squared_distances / (2 * sigma ** 2))
    # weights = weights * T.unsqueeze(-1)

    return weights

def fill_sim_mtx(weights, idx, fps_num, lengthmap):
    weights = torch.cat([weights[i][:lengthmap[i]] for i in range(weights.shape[0])], dim=0)
    sim_mtx = torch.zeros((lengthmap.shape[0], fps_num, weights.shape[-1]))
    sim_mtx[idx] = weights
    zero_row = sim_mtx.sum(dim=-1) == 0
    nonzero_row = sim_mtx.sum(dim=-1) != 0
    cls = torch.zeros(lengthmap.shape[0], fps_num)
    cls[zero_row] = 1
    # sim_mtx[nonzero_row] = F.softmax(sim_mtx[nonzero_row], dim=-1)
    sim_mtx = torch.cat((sim_mtx, cls.unsqueeze(-1)), dim=-1)
    return sim_mtx


def generate_gt_sim(sample_idx, img_crd, coord, intrinsic, extrinsic, vis=False, x3d=None, sigma=1.0):
    bs = coord.shape[0]
    fps_num = sample_idx[0].shape[-1]
    sample_idx = sample_idx.to(torch.int16)
    # extrinsic = torch.from_numpy(quat2mat(extrinsic)).to(torch.float32)
    img_crd = img_crd[0]
    img_crd_pos = img_crd[..., :3]
    img_crd_t = img_crd[..., -2]
    img_crd_idx = img_crd[..., -1].to(torch.int16)
    # idx[i][0]: index in img_crd voxel, idx[i][1]: index in sample_idx
    matches = (img_crd_idx.unsqueeze(-1) == sample_idx.unsqueeze(1))
    idx = matches.nonzero(as_tuple=True)
    idx_crd = idx[0], idx[1]
    idx_sample = idx[0], idx[2]

    lengthmap = torch.unique_consecutive(idx[0], return_counts=True)[1]
    img_crd_t = pad_sequence(list(torch.split(img_crd_t[idx_crd], lengthmap.tolist())), batch_first=True, padding_value=0)
    x3d_sample = pad_sequence(list(torch.split(img_crd_pos[idx_crd], lengthmap.tolist())), batch_first=True, padding_value=0).to(torch.float32)

    x2d_proj = batch_projection_2d(x3d_sample, intrinsic, extrinsic) # y, x
    batch_weights_2d = reproj_gaussian_weight(coord, x2d_proj, img_crd_t, sigma)
    # batch_weights_2d = reproj_bilinear_weights(coord, x2d_proj, img_crd_t)
    gt_sim = fill_sim_mtx(batch_weights_2d, idx_sample, fps_num, lengthmap)
    if vis:
        x2d_proj = batch_projection_2d(x3d, intrinsic, extrinsic)
        return gt_sim, coord, x2d_proj
    return gt_sim


def sample_ref_feat(ref_rgbs, inv_crd, encoder, ref_intr, ref_pose, voxel, use_nerf_mlp, res=32):
    # define shape
    bs = ref_rgbs.shape[0]
    img_sample = ref_rgbs.shape[1]
    voxel_sample = voxel.shape[1]

    # extract feature maps
    ref_rgbs = ref_rgbs.view(bs * img_sample, ref_rgbs.shape[2], ref_rgbs.shape[3], ref_rgbs.shape[4]).permute(0, 3, 1, 2)
    featmaps = encoder(ref_rgbs)

    # batch back projection on all multi-view images for every voxel
    x2d_proj = multiview_back_projection(voxel, ref_intr, ref_pose).permute(0, 2, 1, 3)
    x2d_proj = normalize(x2d_proj, res).reshape(bs * img_sample, voxel_sample, 1, 2)

    # feature sampling
    ref_feat = F.grid_sample(featmaps, x2d_proj, align_corners=True, padding_mode='border')
    ref_feat = ref_feat.squeeze().permute(0, 2, 1).view(bs, img_sample, voxel_sample, -1).permute(0, 2, 1, 3)
    ref_feat = ref_feat.gather(2, inv_crd[..., 0].to(torch.int64).unsqueeze(-1).expand(-1, -1, -1, ref_feat.shape[-1]))

    if not use_nerf_mlp:
        return ref_feat, featmaps
    else:
        # downsample reference rgb to the same resolution as featmap
        ds_size = featmaps.shape[-2:]
        ref_rgbs = F.interpolate(ref_rgbs, size=ds_size, mode='bilinear', align_corners=False)

        # rgb sampling
        ref_rgb = F.grid_sample(ref_rgbs, x2d_proj, align_corners=True, padding_mode='border')
        ref_rgb = ref_rgb.squeeze().permute(0, 2, 1).view(bs, img_sample, voxel_sample, -1).permute(0, 2, 1, 3)
        ref_rgb = ref_rgb.gather(2, inv_crd[..., 0].to(torch.int64).unsqueeze(-1).expand(-1, -1, -1, ref_rgb.shape[-1]))

        # fuse feature
        fused_feat = torch.cat((ref_rgb, ref_feat), dim=-1)
        return fused_feat, featmaps


def vis_gt_sim(sim_pred, sim_gt, x2d, x2d_proj, save_dir=None, res=32, gt=False):
    high_res = res * 2
    heatmap = torch.zeros((sim_gt.shape[0], res, res))
    heatmap = heatmap + 0.5
    x = x2d[:, 0].to(torch.long)
    y = x2d[:, 1].to(torch.long)
    # length = x.shape[0]
    # sim_gt = sim_gt[:, :length]
    # sim_pred = sim_pred[:, :length]
    heatmap = heatmap.unsqueeze(-1).repeat(1, 1, 1, 3)
    for i in range(sim_gt.shape[0]):
        # make sim_gt green
        # gt_color = torch.tensor([1, 0, 1]).to(torch.float32) * sim_gt[i].unsqueeze(-1)
        pred_color = torch.tensor([1, 1, 0]).to(torch.float32) * sim_pred[i].unsqueeze(-1)

        heatmap[i, x, y] = 1 - pred_color
    # heatmap[x, y] = sim_aggre[:length]
    rgb_heatmap = F.interpolate(heatmap.permute(0, 3, 1, 2), size=high_res, mode='nearest').permute(0, 2, 3, 1)

    for i in range(sim_gt.shape[0]):
        heatmap_single = rgb_heatmap[i]
        x_proj = (((x2d_proj[i, 0] + 0.5) / res) * high_res).to(torch.long)
        y_proj = (((x2d_proj[i, 1] + 0.5) / res) * high_res).to(torch.long)
        # make x2d_proj green
        if sim_gt[i].sum(-1) == 0:
            heatmap_single[x_proj, y_proj] = torch.tensor([1, 0, 0]).to(torch.float32)
        else:
            heatmap_single[x_proj, y_proj] = torch.tensor([0, 1, 0]).to(torch.float32)

        if save_dir is not None:
            plt.imshow(heatmap_single.numpy(), cmap='gray', vmin=0, vmax=1)
            plt.savefig(os.path.join(save_dir, f'heatmap_{i}.png'))

        plt.close()

def vis_sim(sim_gt, x2d, x2d_proj, save_dir=None, res=32, gt=False):
    high_res = res * 8
    heatmap = torch.zeros((sim_gt.shape[0], res, res))
    heatmap = heatmap + 0.5
    x = x2d[:, 0].to(torch.long)
    y = x2d[:, 1].to(torch.long)
    # length = x.shape[0]
    # sim_gt = sim_gt[:, :length]
    # sim_pred = sim_pred[:, :length]
    heatmap = heatmap.unsqueeze(-1).repeat(1, 1, 1, 3)
    for i in range(sim_gt.shape[0]):
        # make sim_gt green
        gt_color = torch.tensor([1, 0, 1]).to(torch.float32) * sim_gt[i].unsqueeze(-1)
        heatmap[i, x, y] = 1 - gt_color

    # heatmap[x, y] = sim_aggre[:length]
    rgb_heatmap = F.interpolate(heatmap.permute(0, 3, 1, 2), size=high_res, mode='nearest').permute(0, 2, 3, 1)
    for i in range(sim_gt.shape[0]):
        heatmap_single = rgb_heatmap[i]
        x_proj = (((x2d_proj[i, 0] + 0.5) / res) * high_res).to(torch.long)
        y_proj = (((x2d_proj[i, 1] + 0.5) / res) * high_res).to(torch.long)
        heatmap_single[x_proj, y_proj] = torch.tensor([0, 0, 0]).to(torch.float32)
        plt.imshow(heatmap_single.numpy(), cmap='gray', vmin=0, vmax=1)
        # plt.savefig(os.path.join(save_dir, f'heatmap_{i}.png'))
        plt.show()
        plt.close()


def sample_from_img_crd(img_crd, num=50):
    lengthmap = img_crd[1]
    img_crd = img_crd[0][..., -1].to(torch.int16)
    sample_voxels = []
    for i, length in enumerate(lengthmap):
        idx = torch.randperm(length)[:num]#.unsqueeze(-1)
        voxel_idx = torch.gather(img_crd[i, :length], 0, idx)
        sample_voxels.append(voxel_idx)
    return torch.stack(sample_voxels)


def sample_from_all_matches(matches, voxel_samples, sampling_num):
    matches = matches.view(-1, sampling_num)
    # good_matches = matches >= 2
    samples = []
    for i in range(matches.shape[0]):
        valid_id = (matches[i] >= 2).nonzero(as_tuple=True)[0]
        assert len(valid_id) >= voxel_samples
        sampled_id = valid_id[torch.randperm(len(valid_id))[:voxel_samples]]
        samples.append(sampled_id)
    return torch.stack(samples)


def dual_softmax(matrix):
    row_sfmx = F.softmax(matrix, dim=2)
    col_sfmx = F.softmax(matrix, dim=1)
    return row_sfmx * col_sfmx