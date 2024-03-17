import torch
import torch.nn as nn
import os
from src.models.ibrnet.mlp_network import IBRNet
from src.models.ibrnet.featnet_resfpn import ResNet
from src.models.ibrnet.featnet_resfpn import BasicBlock
from src.models.ibrnet.featnet_resfpn import Bottleneck

from src.models.ibrnet.render_ray import render_rays
from src.models.ibrnet.render_image import render_image, project_image
from src.models.ibrnet.sample_ray import RaySamplerSingleImage
from src.models.ibrnet.criterion import Criterion
from src.models.ibrnet.projection import Projector


class IBRNetModel(nn.Module):
    def __init__(self, args, ibrnet, feature_net):
        super().__init__()
        self.args = args
        self.ibrnet = ibrnet
        self.feature_net = feature_net
        self.projector = Projector()

    def forward(self, batch, device='cpu', mode='image'):
        batch.update_device(device)
        if self.training:
            rays = batch.random_sample(self.args.N_rand)
            featmap = self.feature_net(rays['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
            ret = render_rays(rays=rays,
                              model=self.ibrnet,
                              featmaps=featmap,
                              projector=self.projector,
                              N_samples=self.args.N_samples,
                              inv_uniform=self.args.inv_uniform,
                              white_bkgd=self.args.white_bkgd)
        else:
            rays = batch.get_all()
            if mode == 'image':
                featmap = self.feature_net(rays['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
                ret = render_image(rays=rays,
                                   model=self.ibrnet,
                                   projector=self.projector,
                                   chunk_size=self.args.N_rand,
                                   N_samples=self.args.N_samples,
                                   inv_uniform=self.args.inv_uniform,
                                   white_bkgd=self.args.white_bkgd,
                                   featmap=featmap)
                ret = batch.fill_img(ret)
            if mode == 'voxel':
                featmap = self.feature_net(rays['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
                ret = project_image(rays=rays,
                                    model=self.ibrnet,
                                    projector=self.projector,
                                    chunk_size=self.args.N_rand,
                                    N_samples=self.args.N_samples,
                                    inv_uniform=self.args.inv_uniform,
                                    white_bkgd=self.args.white_bkgd,
                                    featmap=featmap)
        return ret



