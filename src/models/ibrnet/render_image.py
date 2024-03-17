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


import torch
from collections import OrderedDict
from src.models.ibrnet.render_ray import render_rays, project_rays
import gc


def render_single_image(ray_sampler,
                        ray_batch,
                        model,
                        projector,
                        chunk_size,
                        N_samples,
                        inv_uniform=False,
                        N_importance=0,
                        det=False,
                        white_bkgd=False,
                        render_stride=1,
                        featmaps=None):
    '''
    :param ray_sampler: RaySamplingSingleImage for this view
    :param model:  {'net_coarse': , 'net_fine': , ...}
    :param chunk_size: number of rays in a chunk
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :return: {'outputs_coarse': {'rgb': numpy, 'depth': numpy, ...}, 'outputs_fine': {}}
    '''

    all_ret = OrderedDict([('outputs_coarse', OrderedDict()),
                           ('outputs_fine', OrderedDict())])

    N_rays = ray_batch['ray_o'].shape[0]

    for i in range(0, N_rays, chunk_size):
        chunk = OrderedDict()
        for k in ray_batch:
            if k in ['camera', 'depth_range', 'src_rgbs', 'src_cameras']:
                chunk[k] = ray_batch[k]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i:i+chunk_size]
            else:
                chunk[k] = None

        ret = render_rays(chunk, model, featmaps,
                          projector=projector,
                          N_samples=N_samples,
                          inv_uniform=inv_uniform,
                          N_importance=N_importance,
                          det=det,
                          white_bkgd=white_bkgd)

        # handle both coarse and fine outputs
        # cache chunk results on cpu
        if i == 0:
            for k in ret['outputs_coarse']:
                all_ret['outputs_coarse'][k] = []

            if ret['outputs_fine'] is None:
                all_ret['outputs_fine'] = None
            else:
                for k in ret['outputs_fine']:
                    all_ret['outputs_fine'][k] = []

        for k in ret['outputs_coarse']:
            all_ret['outputs_coarse'][k].append(ret['outputs_coarse'][k].cpu())

        if ret['outputs_fine'] is not None:
            for k in ret['outputs_fine']:
                all_ret['outputs_fine'][k].append(ret['outputs_fine'][k].cpu())

    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]
    # merge chunk results and reshape
    for k in all_ret['outputs_coarse']:
        if k == 'random_sigma':
            continue
        tmp = torch.cat(all_ret['outputs_coarse'][k], dim=0).reshape((rgb_strided.shape[0],
                                                                      rgb_strided.shape[1], -1))
        all_ret['outputs_coarse'][k] = tmp.squeeze()

    all_ret['outputs_coarse']['rgb'][all_ret['outputs_coarse']['mask'] == 0] = 1.
    if all_ret['outputs_fine'] is not None:
        for k in all_ret['outputs_fine']:
            if k == 'random_sigma':
                continue
            tmp = torch.cat(all_ret['outputs_fine'][k], dim=0).reshape((rgb_strided.shape[0],
                                                                        rgb_strided.shape[1], -1))

            all_ret['outputs_fine'][k] = tmp.squeeze()

    return all_ret

def render_image(rays,
                 model,
                 projector,
                 chunk_size,
                 N_samples,
                 inv_uniform=False,
                 white_bkgd=False,
                 render_stride=1,
                 featmap=None):

    N_rays = rays['ray_o'].shape[0]
    rendered_img = {}
    for i in range(0, N_rays, chunk_size):
        ray_chunk = OrderedDict()
        for k in rays:
            if k in ['ray_o', 'ray_d', 'rgb']:
                ray_chunk[k] = rays[k][i:i+chunk_size]
            elif rays[k] is not None:
                ray_chunk[k] = rays[k]

        ret = render_rays(rays=ray_chunk,
                          model=model,
                          featmaps=featmap,
                          projector=projector,
                          N_samples=N_samples,
                          inv_uniform=inv_uniform,
                          white_bkgd=white_bkgd)

        if i == 0:
            for k in ret:
                if k == 'rgb' or k == 'depth' or k == 'gt_rgb':
                    rendered_img[k] = []
        for k in ret:
            if k == 'rgb' or k == 'depth' or k == 'gt_rgb':
                rendered_img[k].append(ret[k].cpu())
    for k in rendered_img:
        rendered_img[k] = torch.cat(rendered_img[k], dim=0)

    return rendered_img


def project_image(rays,
                  model,
                  projector,
                  chunk_size,
                  N_samples,
                  inv_uniform=False,
                  white_bkgd=False,
                  render_stride=1,
                  featmap=None):
    projected_image = {}
    projected_image['pts'] = []
    projected_image['sigma'] = []
    projected_image['transparency'] = []
    projected_image['pixels'] = []
    projected_image['depth'] = []
    projected_image['ray_d'] = []
    projected_image['ray_o'] = []
    projected_image['weights'] = []

    N_rays = rays['ray_o'].shape[0]
    for i in range(0, N_rays, chunk_size):
        ray_chunk = OrderedDict()
        for k in rays:
            if k in ['ray_o', 'ray_d', 'rgb']:
                ray_chunk[k] = rays[k][i:i+chunk_size]
            elif rays[k] is not None:
                ray_chunk[k] = rays[k]

        ret = project_rays(rays=ray_chunk,
                           model=model,
                           featmaps=featmap,
                           projector=projector,
                           N_samples=N_samples,
                           inv_uniform=inv_uniform,
                           white_bkgd=white_bkgd)

        projected_image['pts'].append(ret[0])
        projected_image['sigma'].append(ret[1])
        projected_image['transparency'].append(ret[2])
        projected_image['pixels'].append(ret[3])
        projected_image['depth'].append(ret[4])
        projected_image['ray_d'].append(ret[5])
        projected_image['ray_o'].append(ret[6])
        projected_image['weights'].append(ret[7])
    for k in projected_image:
        if len(projected_image[k]) == 0:
            x=1
        projected_image[k] = torch.cat(projected_image[k], dim=0)
    return projected_image


