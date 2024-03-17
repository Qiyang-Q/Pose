# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2


rng = np.random.RandomState(234)

########################################################################################################################
# ray batch sampling
########################################################################################################################
def vis_img(img):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect='equal')
    plt.imsave('test.png', img)

def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return W, H, intrinsics, c2w


def dilate_img(img, kernel_size=20):
    import cv2
    assert img.dtype == np.uint8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(img / 255, kernel, iterations=1) * 255
    return dilation


def post_process_mask(mask):
    mask = mask.numpy()
    mask = mask[0][:, :, None]
    kernel = np.ones((3, 3), np.uint8)
    mask_refine = cv2.erode(mask, kernel, iterations=1)
    return torch.from_numpy(mask_refine).unsqueeze(0)

class RaySamplerSingleImage(object):
    def __init__(self, data, resize_factor=1, render_stride=1, voxelize=False):
        super().__init__()
        self.render_stride = render_stride
        self.rgb = data['rgb']
        self.mask = data['mask']
        # if voxelize:
        #     self.mask = post_process_mask(self.mask)
        self.rgb_norm = data['rgb_norm']
        # self.mask = data['mask'] if 'mask' in data.keys() else None
        self.depth_range = data['depth_range']
        self.intrin = data['render_intrin']
        self.c2w_mat = data['render_cam']
        self.obj_pose = data['render_obj'].to(torch.float32)
        self.density_offset = data['density_offset']
        self.H, self.W = self.rgb.shape[1:3]
        self.device = self.rgb_norm.device
        # W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
        self.batch_size = len(self.rgb)
        # half-resolution output
        if resize_factor != 1:
            self.W = int(self.W * resize_factor)
            self.H = int(self.H * resize_factor)
            self.intrin[:, :2, :3] *= resize_factor
            self.rgb = F.interpolate(self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(0, 2, 3, 1)
            # if self.mask is not None:
            #     self.mask = F.interpolate(self.mask.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(0, 2, 3, 1)
        self.rays_o, self.rays_d, self.pixels = self.get_rays_single_image(self.H, self.W, self.intrin, self.c2w_mat)
        self.rgb = self.rgb.reshape(-1, 3)
        self.mask = self.mask.squeeze().reshape(-1)
        self.rgb = self.rgb[self.mask]
        self.single_view = False
        # if self.rgb_norm is not None:
        #     self.rgb_norm = self.rgb_norm.reshape(-1, 3)
        # if self.mask is not None:
        #     self.mask = self.mask.reshape(-1)


        if 'src_rgbs' in data.keys():
            self.src_rgbs = data['src_rgbs']
            if self.src_rgbs.shape[1] == 1:
                self.single_view = True
        else:
            self.src_rgbs = None
        if 'src_cam' in data.keys():
            self.src_cam = data['src_cam']
            self.src_intrin = data['src_intrin']
        else:
            self.src_cam = None
            self.src_intrin = None



    def get_rays_single_image(self, H, W, intrinsics, c2w):
        '''
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        '''
        u, v = np.meshgrid(np.arange(W)[::self.render_stride], np.arange(H)[::self.render_stride])
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels)
        mask = self.mask.squeeze().reshape(-1)
        pixels = pixels[:, mask]
        batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)
        pixels = pixels.permute(1, 0)[:, :2]
        rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)  # B x HW x 3
        return rays_o, rays_d, pixels

    def get_all(self):
        ret = {'ray_o': self.rays_o.to(self.device, non_blocking=True),
               'ray_d': self.rays_d.to(self.device, non_blocking=True),
               'pixels': self.pixels.to(self.device, non_blocking=True),
               'intrin': self.intrin.to(self.device, non_blocking=True),
               'extrin': self.c2w_mat.to(self.device, non_blocking=True),
               'depth_range': self.depth_range.to(self.device, non_blocking=True),
               'rgb': self.rgb.to(self.device, non_blocking=True) if self.rgb is not None else None,
               'rgb_norm': self.rgb_norm.to(self.device, non_blocking=True) if self.rgb_norm is not None else None,
               'src_rgbs': self.src_rgbs.to(self.device, non_blocking=True) if self.src_rgbs is not None else None,
               'src_cam': self.src_cam.to(self.device, non_blocking=True) if self.src_cam is not None else None,
               'src_intrin': self.src_intrin.to(self.device, non_blocking=True) if self.src_intrin is not None else None,
               'intrin': self.intrin.to(self.device, non_blocking=True),
               'obj_pose': self.obj_pose.to(self.device, non_blocking=True) if self.obj_pose is not None else None,
               }
        return ret

    def get_masked(self):
        rays_o = self.rays_o
        rays_d = self.rays_d
        mask_idx = self.mask.nonzero().squeeze()
        rays_o = rays_o[mask_idx]
        rays_d = rays_d[mask_idx]
        pixels = self.pixels[mask_idx]
        ret = {'ray_o': rays_o.to(self.device),
               'ray_d': rays_d.to(self.device),
               'depth_range': self.depth_range.to(self.device),
               'camera': self.camera.to(self.device),
               'pixels': pixels.to(self.device),
               'rgb': self.rgb.to(self.device) if self.rgb is not None else None,
               'rgb_norm': self.rgb_norm.to(self.device) if self.rgb_norm is not None else None,
               'src_rgbs': self.src_rgbs.to(self.device) if self.src_rgbs is not None else None,
               'src_rgbs_norm': self.src_rgbs_norm.to(self.device) if self.src_rgbs_norm is not None else None,
               'src_cameras': self.src_cam.to(self.device) if self.src_cam is not None else None,
               'intrin': self.intrin.to(self.device, non_blocking=True),
               'obj_pose': self.obj_pose.to(self.device, non_blocking=True) if self.obj_pose is not None else None,
               }
        return ret


    def sample_random_pixel(self, N_rand):
        if N_rand < len(self.pixels):
            select_inds = rng.choice(len(self.pixels), size=(N_rand,), replace=False)
        else:
            select_inds = np.arange(len(self.pixels))
        return select_inds

    def random_sample(self, N_rand):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''

        select_inds = self.sample_random_pixel(N_rand)

        rays_o = self.rays_o[select_inds]
        rays_d = self.rays_d[select_inds]
        pixels = self.pixels[select_inds]

        rgb = self.rgb[select_inds]

        ret = {'ray_o': rays_o.to(self.device, non_blocking=True),
               'ray_d': rays_d.to(self.device, non_blocking=True),
               'pixels': pixels.to(self.device, non_blocking=True),
               'intrin': self.intrin.to(self.device, non_blocking=True),
               'extrin': self.c2w_mat.to(self.device, non_blocking=True),
               'depth_range': self.depth_range.to(self.device, non_blocking=True),
               'rgb': rgb.to(self.device, non_blocking=True) if rgb is not None else None,
               'mask': self.mask.to(self.device, non_blocking=True) if self.mask is not None else None,
               'rgb_norm': self.rgb_norm.to(self.device, non_blocking=True) if self.rgb_norm is not None else None,
               'src_rgbs': self.src_rgbs.to(self.device, non_blocking=True) if self.src_rgbs is not None else None,
               'src_cam': self.src_cam.to(self.device, non_blocking=True) if self.src_cam is not None else None,
               'src_intrin': self.src_intrin.to(self.device, non_blocking=True) if self.src_intrin is not None else None,
               'selected_inds': select_inds,
               'obj_pose': self.obj_pose.to(self.device, non_blocking=True) if self.obj_pose is not None else None,
               'single_view': self.single_view
        }
        # ret = {'ray_o': rays_o,
        #        'ray_d': rays_d,
        #        'intrin': self.intrin,
        #        'extrin': self.c2w_mat,
        #        'depth_range': self.depth_range,
        #        'rgb': rgb if rgb is not None else None,
        #        'rgb_norm': self.rgb_norm if self.rgb_norm is not None else None,
        #        'src_rgbs': self.src_rgbs if self.src_rgbs is not None else None,
        #        'src_cam': self.src_cam if self.src_cam is not None else None,
        #        'src_intrin': self.src_intrin if self.src_intrin is not None else None,
        #        'selected_inds': select_inds
        # }
        return ret

    def update_device(self, device):
        self.device = device


    def get_density_offset(self):
        return self.density_offset.to(torch.float32)

    def fill_img(self, ret):
        rend_pixel = ret['rgb']
        rend_depth = torch.zeros(self.H * self.W)
        rend_img = torch.zeros(self.H * self.W, 3)
        rend_img[self.mask] = rend_pixel
        rend_depth[self.mask] = ret['depth']
        rend_img = rend_img.reshape(self.H, self.W, 3)
        rend_depth = rend_depth.reshape(self.H, self.W)
        return {'rgb': rend_img, 'depth': rend_depth}
