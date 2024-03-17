import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.ibrnet.mlp_network
from src.models.matchers.positional_encoding import PositionalEncoding, PositionEncodingSine, KeypointEncoding_linear
from src.utils.pose_utils import *
from src.models.ibrnet.mlp_network import DenseLayer
from src.utils.debug_utils import *


class pose_solver(nn.Module):
    def __init__(self, pnp_solver):
        super().__init__()
        self.pnp_solver = pnp_solver

    def forward(self, ret):
        x3d = ret['x3d'].unsqueeze(0)
        x2d = ret['pixels'].clone().to(x3d.device).unsqueeze(0)
        w2d = torch.ones_like(x2d).to(x3d.device)
        intrinsic = ret['intrin'].to(x3d.device)[:, :3, :3]
        pose_gt = ret['gt_pose'].to(x3d.device)
        pose_pred, logweights, cost = self.pnp_solver(x3d, x2d, w2d, intrinsic, pose_gt, res=128)
        return pose_pred, logweights, cost


