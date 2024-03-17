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


def extract_surface(pos):
    pos = pos.cpu().numpy().astype(np.uint8)
    pos_x_max, pos_x_min = int(max(pos[:, 0])), int(min(pos[:, 0]))
    pos_y_max, pos_y_min = int(max(pos[:, 1])), int(min(pos[:, 1]))
    pos_z_max, pos_z_min = int(max(pos[:, 2])), int(min(pos[:, 2]))
    volume = np.zeros((pos_x_max - pos_x_min + 3, pos_y_max - pos_y_min + 3, pos_z_max - pos_z_min + 3))
    volume[pos[:, 0] - pos_x_min + 1, pos[:, 1] - pos_y_min + 1, pos[:, 2] - pos_z_min + 1] = 1

    surface, _, _, _ = measure.marching_cubes(volume, level=0.5, method='lewiner')
    surface1 = np.ceil(surface).astype(np.uint8)
    surface2 = np.floor(surface).astype(np.uint8)
    surface = np.unique(np.concatenate((surface1, surface2), axis=0), axis=0)
    surface[:, 0] = surface[:, 0] + pos_x_min - 1
    surface[:, 1] = surface[:, 1] + pos_y_min - 1
    surface[:, 2] = surface[:, 2] + pos_z_min - 1

    return torch.from_numpy((pos == surface[:, None]).all(axis=2).any(axis=0)).to('cuda')


def vis_voxel(voxel, with_border=False):
    vis_voxel = voxel.cpu().numpy()
    pcd_voxel = o3d.geometry.PointCloud()
    pcd_voxel.points = o3d.utility.Vector3dVector(vis_voxel)
    pcd_voxel.paint_uniform_color([0.5, 0.5, 0.5])
    # o3d.visualization.draw_geometries([pcd_voxel])
    if with_border:
        size = 50
        bounds = np.array([[0, 0, 0], [0, 0, size], [0, size, 0], [0, size, size], [size, 0, 0], [size, 0, size], [size, size, 0], [size, size, size]])
        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bounds)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd_voxel, line_set])
    else:
        o3d.visualization.draw_geometries([pcd_voxel])


def vis_ray(ray, voxel):
    vis_ray = ray.cpu().numpy()
    pcd_ray = o3d.geometry.PointCloud()
    pcd_ray.points = o3d.utility.Vector3dVector(vis_ray)
    pcd_ray.paint_uniform_color([0.9, 0.9, 0.9])

    vis_voxel = voxel.cpu().numpy()
    pcd_voxel = o3d.geometry.PointCloud()
    pcd_voxel.points = o3d.utility.Vector3dVector(vis_voxel)
    pcd_voxel.paint_uniform_color([0.5, 0.5, 0.5])

    o3d.visualization.draw_geometries([pcd_ray, pcd_voxel])



class Voxelizer:

    def __init__(self, dataset, renderer, device):
        self.device = f'cuda:{device}'
        self.voxel_info, self.data_idx = self.init_voxels(dataset)
        self.renderer = renderer
        self.sequential_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        self.accu_pos = []
        self.accu_sigma = []
        self.accu_transparency = []
        self.accu_weights = []
        self.accu_surface = []
        self.accu_sigma_w = []
        self.correspondence = {}
        self.proj_stride = 1
        self.transp_thresh = 0.8
        self.opacity_thresh = 0.99
        self.density_thresh = 0.5

    def init_voxels(self, dataset):
        voxel_info = {}
        data = dataset.data
        data_idx = dataset.data_idx
        for catagory_name in data.keys():
            voxel_info[catagory_name] = {}
            radius = data[catagory_name]['radius']
            index_range = data[catagory_name]['index_range']
            diameter = radius * 2
            scaling = 1.0
            whole_voxel_size = torch.tensor([diameter, diameter, diameter]) * scaling
            grid_counts = torch.tensor([50, 50, 50])
            grid_size = whole_voxel_size / grid_counts
            voxel_center = whole_voxel_size / 2
            voxel_info[catagory_name]['index_range'] = index_range
            voxel_info[catagory_name]['grid_size'] = grid_size.to(self.device)
            voxel_info[catagory_name]['voxel_center'] = voxel_center.to(self.device)
            voxel_info[catagory_name]['grid_counts'] = grid_counts.to(self.device)
            voxel_info[catagory_name]['voxel_size'] = whole_voxel_size
            voxel_info[catagory_name]['data_size'] = index_range[1] - index_range[0]
            voxel_info[catagory_name]['save_path'] = os.path.join(dataset.dataset_dirs, catagory_name)
        return voxel_info, data_idx

    def voxelize(self):
        print('Start voxelizing')
        self.renderer.eval()
        end = False
        pbar_outside = _tqdm(total=len(self.voxel_info.keys()), leave=False, position=0, file=sys.stdout)
        pbar_inside = self.init_pbar(0)
        dataset_length = len(self.data_idx)
        with torch.no_grad():
            for idx, batch in enumerate(self.sequential_loader):
                ray_batch = RaySamplerSingleImage(batch, voxelize=True)
                density_offset = ray_batch.get_density_offset().to(self.device)
                if idx % self.proj_stride != 0:
                    continue
                projected_image = self.renderer(ray_batch, device=self.device, mode='voxel')
                self.update_voxel(projected_image, idx, density_offset)
                pbar_inside.update(self.proj_stride)
                if idx + self.proj_stride >= dataset_length:
                    generate = True
                    end = True
                else:
                    generate = self.data_idx[idx]['catagory'] != self.data_idx[idx + self.proj_stride]['catagory']
                if generate:
                    self.generate_voxel(self.data_idx[idx]['catagory'])
                    self.save_voxel(self.voxel_info[self.data_idx[idx]['catagory']]['save_path'])
                    self.clear_cache()
                    if end:
                        pbar_inside.close()
                        pbar_outside.close()
                    else:
                        pbar_outside.update(1)
                        pbar_inside.close()
                        pbar_inside = self.init_pbar(idx)

    def update_voxel(self, projected_image, idx, density_offset=1):
        '''
        :param xyz: [N_rays, N_samples, 3]
        :param sigma: [N_rays, N_samples], density of each sample on the ray
        :param transparency: [N_rays, N_samples], transparency of each sample on the ray
        :param feat: [N_rays, N_samples, feat_channel], 2D feature of each sample on the ray
        :return a voxel of shape [Length, Width, Height] with density
        '''
        catagory_name = self.data_idx[idx]['catagory']
        img_id = self.data_idx[idx]['img_id']
        center = self.voxel_info[catagory_name]['voxel_center']
        grid_size = self.voxel_info[catagory_name]['grid_size']
        grid_counts = self.voxel_info[catagory_name]['grid_counts']

        xyz = projected_image['pts']
        sigma = projected_image['sigma']
        transparency = projected_image['transparency']
        z_vals = projected_image['depth']
        ray_d = projected_image['ray_d']
        ray_o = projected_image['ray_o']
        weights = projected_image['weights']

        weights_test = weights
        weights_test[weights_test < 1e-4] = 0
        depth = torch.sum(weights_test * z_vals, dim=-1)
        weighted_sigma = torch.sum(weights * sigma, dim=-1)
        surface = ray_o + ray_d * depth.unsqueeze(-1)
        surface = ((surface + center) // grid_size).to(torch.int8)

        if img_id not in self.correspondence:
            self.correspondence[img_id] = {'rays': None, 'weights': None, 'transparency': None}
        xyz = ((xyz + center) // grid_size).to(torch.int8)
        self.correspondence[img_id]['rays'] = xyz
        self.correspondence[img_id]['weights'] = weights_test
        self.correspondence[img_id]['transparency'] = transparency
        # sigma = sigma.view(-1)
        # T = transparency.view(-1)
        xyz = xyz.view(-1, 3)
        weights = weights.view(-1)

        nonzero_ind = (weights > EPS).nonzero().squeeze()
        weights = weights[nonzero_ind]
        # sigma = sigma[nonzero_ind]
        xyz = xyz[nonzero_ind]
        # T = T[nonzero_ind]

        gc = grid_counts
        conds = [surface[:, 0] < 0, surface[:, 0] > gc[0], surface[:, 1] < 0, surface[:, 1] > gc[1], surface[:, 2] < 0, surface[:, 2] > gc[2]]
        oob_cond = conds[0]
        for i in range(1, len(conds)):
            oob_cond = torch.logical_or(oob_cond, conds[i])
        surface = surface[~oob_cond]
        # sigma = sigma[~oob_cond]
        # T = T[~oob_cond]
        weighted_sigma = weighted_sigma[~oob_cond]


        # remove duplicate points
        xyz, ind, count = torch.unique(xyz, dim=0, return_inverse=True, return_counts=True, sorted=False)
        # sigma = torch.bincount(ind, weights=sigma, minlength=xyz.shape[0]) / count
        weights = self.max_in_bin(ind, weights) * density_offset
        # T = torch.bincount(ind, weights=T, minlength=xyz_u.shape[0]) / count
        surface, ind, count = torch.unique(surface, dim=0, return_inverse=True, return_counts=True, sorted=False)
        weighted_sigma = torch.bincount(ind, weights=weighted_sigma, minlength=surface.shape[0]) / count
        weighted_sigma = weighted_sigma * density_offset.sqrt()

        self.accu_pos.append(xyz)
        # self.accu_sigma.append(sigma)
        # self.accu_transparency.append(T)
        self.accu_weights.append(weights)
        self.accu_surface.append(surface)
        self.accu_sigma_w.append(weighted_sigma)

    def vis_voxel(self):
        print('start to visualize')
        vis = self.voxel_pos.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([pcd])

    def vis_vxl(self, pos):
        print('start to visualize')
        vis = pos.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([pcd])

    def clear_cache(self):
        self.accu_pos = []
        self.accu_sigma = []
        self.accu_transparency = []
        self.accu_weights = []
        self.accu_surface = []
        self.accu_sigma_w = []
        self.correspondence = {}
        torch.cuda.empty_cache()

    # doesn't work in OnePose
    # def filter_voxel(self, pos, density=None, T=None, weights=None):
    #     pos, ind, count = torch.unique(pos, dim=0, return_inverse=True, return_counts=True)
    #     ###### Filter by transparency threshold ######
    #     # mask = T > self.transp_thresh
    #     # true_counts = torch.bincount(ind[mask], minlength=ind.max().item() + 1)
    #     # total_counts = torch.bincount(ind, minlength=ind.max().item() + 1)
    #     # valid_voxels = torch.nonzero(true_counts == total_counts, as_tuple=False).squeeze()
    #     # return pos[valid_voxels]
    #     ###### Filter by opacity threshold ######
    #     # density = torch.bincount(ind, weights=density, minlength=pos.shape[0])# / 377
    #     # opacity = 1 - torch.exp(-density)
    #     # pos = pos[opacity > self.opacity_thresh]
    #     # return pos
    #     # ###### Filter by density threshold ######
    #     # mask = density > self.density_thresh
    #     # true_counts = torch.bincount(ind[mask], minlength=ind.max().item() + 1)
    #     # total_counts = torch.bincount(ind, minlength=ind.max().item() + 1)
    #     # valid_voxels = torch.nonzero(true_counts == total_counts, as_tuple=False).squeeze()
    #     # return pos[valid_voxels]
    #     ##### Filter by weights threshold ######
    #     # weights = self.min_in_bin(ind, weights)
    #     weights = torch.bincount(ind, weights=weights, minlength=pos.shape[0])
    #     # mean_weights = weights.mean()
    #     # std_weights = weights.std()
    #     # vis_voxel(pos)
    #     threshold = weights.quantile(0.95)
    #     pos = pos[weights > threshold]
    #     # vis_voxel(pos)
    #     pos = self.remove_outlier(pos, iter=5)
    #     return pos

    def generate_voxel(self, catagory_name):
        print('start to generate voxel')
        grid_size = self.voxel_info[catagory_name]['grid_size']
        center = self.voxel_info[catagory_name]['voxel_center']
        pos = torch.cat(self.accu_pos, dim=0)
        # density = torch.cat(self.accu_sigma, dim=0)
        # T = torch.cat(self.accu_transparency, dim=0)
        weights = torch.cat(self.accu_weights, dim=0)

        # pos = self.filter_voxel(pos, weights=weights)
        # vis_voxel(pos)
        pos = self.extract_surface()

        self.voxel_pos = pos * grid_size - center

        # Extract surface and correspondence
        inv_crd = []
        pos_surface = []
        surface_idx = []
        for img_id in self.correspondence:
            # separate correspondence of each image into batches to save memory
            img_surface_idx = []
            img_surface_T = []
            rays = self.correspondence[img_id]['rays']
            weights = self.correspondence[img_id]['weights']
            transparency = self.correspondence[img_id]['transparency']
            # transparency = self.correspondence[img_id][1]
            step = 500
            for i in range(0, rays.shape[0], step):
                rays_mini = rays[i: i + step]
                weights_mini = weights[i: i + step]
                transparency_mini = transparency[i: i + step]
                batch_matches = torch.nonzero((rays_mini[:, :, None, :] == pos).all(dim=3), as_tuple=True)

                batch_matches_idx = torch.unique(batch_matches[0])
                for idx in batch_matches_idx:
                    ray_match = batch_matches[1][batch_matches[0] == idx][0]
                    voxel_matches = batch_matches[2][batch_matches[0] == idx]
                    weights_match = weights_mini[idx, batch_matches[1][batch_matches[0] == idx]]
                    # if weights_match.sum() < 1e-4:
                    #     continue
                    # else:
                    voxel_matches = voxel_matches[0]
                    t_match = transparency_mini[idx, ray_match]
                    if t_match < 0.1:
                        continue
                    img_surface_idx.append(voxel_matches)
                    img_surface_T.append(t_match)
                    surface_idx.append(voxel_matches)
            img_surface_idx = torch.stack(img_surface_idx)
            img_surface_T = torch.stack(img_surface_T)
            img_surface_idx, unique_idx = torch.unique(img_surface_idx, dim=0, return_inverse=True)
            img_surface_T = torch.bincount(unique_idx, weights=img_surface_T, minlength=img_surface_idx.shape[0]) / torch.bincount(unique_idx, minlength=img_surface_idx.shape[0])
            img_surface = self.voxel_pos[img_surface_idx]
            # if img_id in [326, 327, 328]:
            #     vis_voxel(img_surface)
            img_surface = torch.cat([img_surface, img_surface_T.unsqueeze(-1), img_surface_idx.unsqueeze(-1)], dim=1)
            # img_surface = torch.cat([img_surface, img_surface_idx.unsqueeze(-1)], dim=1)
            self.correspondence[img_id] = img_surface

        surface_idx = torch.stack(surface_idx)
        surface_idx = torch.unique(surface_idx, dim=0, return_inverse=False)
        for img_id in self.correspondence:
            # rectify index of img_surface to match surface_idx using inv_idx
            img_surface = self.correspondence[img_id]
            # find the index of img_surface in surface_idx
            img_surface[:, -1] = (surface_idx[:, None] == img_surface[:, -1]).nonzero()[:, 0]
            self.correspondence[img_id] = img_surface.cpu().numpy()

        pos_surface = self.voxel_pos[surface_idx]
        self.voxel_pos = pos_surface
        # vis_voxel(pos_surface)


    def save_voxel(self, path):
        position_path = os.path.join(path, 'voxel_position_v3.npy')
        correspondence_path = os.path.join(path, 'correspondence_v3.npy')
        np.save(position_path, self.voxel_pos.cpu().numpy())
        print('Position voxel saved to {}'.format(position_path))
        np.save(correspondence_path, self.correspondence)
        print('Correspondence saved to {}'.format(correspondence_path))

    def init_pbar(self, idx):
        total_length = self.voxel_info[self.data_idx[idx + self.proj_stride]['catagory']]['data_size']
        catagory_name = self.data_idx[idx + self.proj_stride]['catagory'].split('/')[-1]
        description = f'Generating voxel for: {catagory_name}'
        pbar = _tqdm(total=total_length, leave=False, position=1, desc=description, file=sys.stdout)
        return pbar

    def max_in_bin(self, ind, val):
        max_val = torch.full((ind.max().item() + 1,), float('-inf'), device=self.device)
        max_val.scatter_reduce_(0, ind, val, reduce='amax')
        return max_val


    def extract_surface(self):
        surface = torch.cat(self.accu_surface, dim=0)
        weighted_sigma = torch.cat(self.accu_sigma_w, dim=0)
        surface, ind = torch.unique(surface, dim=0, return_inverse=True)
        weighted_sigma = torch.bincount(ind, weights=weighted_sigma, minlength=surface.shape[0])# / 377
        opacity = 1 - torch.exp(-weighted_sigma)
        threshold = opacity.quantile(0.99)
        threshold = min(threshold, self.opacity_thresh)
        # vis_voxel(surface, with_border=True)
        surface = surface[opacity > threshold]
        surface = self.remove_outlier(surface, iter=8)
        return surface

    def remove_outlier(self, surface, iter=3):
        surface = self.remove_outlier_once(surface, 10, 70)
        surface = self.remove_outlier_once(surface, 6, 30)
        for i in range(iter):
            surface = self.remove_outlier_once(surface, 3, 9)
        # for i in range(iter):

        return surface

    def remove_outlier_once(self, surface, radius, min_neighbor):
        expanded_surface1 = surface[:, None, :]
        expanded_surface2 = surface[None, :, :]
        distance_matrix = torch.sum(torch.abs(expanded_surface1 - expanded_surface2), dim=2)
        mask = torch.sum(distance_matrix < radius, dim=1) > min_neighbor
        return surface[mask]





