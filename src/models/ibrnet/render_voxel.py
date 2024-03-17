import numpy as np
import torch
from collections import OrderedDict
from src.models.ibrnet.render_ray import render_rays
from src.models.ibrnet.render_ray import fill_voxel
import matplotlib.pyplot as plt


def render_single_voxel(ray_sampler,
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
                        featmaps=None,
                        featmap_self=None,
                        voxel=None
                        ):
    '''
    :param ray_sampler: RaySamplingSingleImage for this view
    :param model:  {'net_coarse': , 'net_fine': , ...}
    :param chunk_size: number of rays in a chunk
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :return: {'outputs_coarse': {'rgb': numpy, 'depth': numpy, ...}, 'outputs_fine': {}}
    '''

    device = projector.device
    # all_ret = OrderedDict([('outputs_coarse', OrderedDict()),
    #                        ('outputs_fine', OrderedDict())])
    proj_3d = torch.zeros((0, 3)).to(device)
    proj_3d_idx = torch.zeros((0)).to(device)
    pixel = torch.zeros((0, 2)).to(device)
    N_rays = ray_batch['ray_o'].shape[0]
    ray_batch['rgb'] = ray_batch['rgb']
    img_catagory = ray_sampler.rgb_path[0].split('/')[-3]
    img_id = ray_sampler.rgb_path[0].split('/')[-1].split('.')[0]
    for i in range(0, N_rays, chunk_size):
        chunk = OrderedDict()
        for k in ray_batch:
            if k in ['camera', 'depth_range', 'src_rgbs', 'src_cameras', 'rgb', 'voxel']:
                chunk[k] = ray_batch[k]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i:i+chunk_size]
            else:
                chunk[k] = None

        fill_voxel(chunk, model, featmaps, featmap_self,
                          projector=projector,
                          N_samples=N_samples,
                          inv_uniform=inv_uniform,
                          N_importance=N_importance,
                          det=det,
                          white_bkgd=white_bkgd,
                          voxel=voxel,
                          id=img_id
                         )
    return proj_3d, proj_3d_idx, pixel

