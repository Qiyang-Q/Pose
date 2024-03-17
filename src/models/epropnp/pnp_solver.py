from src.models.epropnp.epropnp import EProPnP6DoF
from src.models.epropnp.levenberg_marquardt import LMSolver, RSLMSolver
from src.models.epropnp.camera import PerspectiveCamera
from src.models.epropnp.cost_fun import AdaptiveHuberPnPCost
import torch.nn as nn
import torch


class pnp_solver(nn.Module):
    def __init__(self, training):
        super().__init__()
        if training:
            self.solver = EProPnP6DoF(mc_samples=512, num_iter=4, normalize=False, solver=LMSolver(dof=6, num_iter=9,
                                        init_solver=RSLMSolver(dof=6, num_points=16, num_proposals=16, num_iter=4)))
        else:
            self.solver = EProPnP6DoF(mc_samples=512, num_iter=4, normalize=False, solver=LMSolver(dof=6, num_iter=12,
                                            init_solver=RSLMSolver(dof=6, num_points=32, num_proposals=32, num_iter=8)))
        self.camera = PerspectiveCamera(z_min=0.01)
        self.cost_fun = AdaptiveHuberPnPCost(relative_delta=0.1)
        self.training = training

    def forward(self, x3d, x2d, w2d, intrinsic, pose_gt, res=64):
        # lb = torch.min(x2d, dim=1)[0]
        # ub = torch.max(x2d, dim=1)[0]
        lb = torch.zeros((len(x2d), 2), device=x3d.device)
        ub = torch.ones((len(x2d), 2), device=x3d.device) * (res - 1)
        self.camera.set_param(intrinsic, lb=lb, ub=ub)
        self.cost_fun.set_param(x2d, w2d)
        if self.training:
            _, _, pose_opt_plus, _, pose_sample_logweights, cost_tgt = self.solver.monte_carlo_forward(
                x3d, x2d, w2d, self.camera, self.cost_fun, pose_init=pose_gt, force_init_solve=True,
                with_pose_opt_plus=True)
            return pose_opt_plus, pose_sample_logweights, cost_tgt
        else:
            pose = self.solver(x3d, x2d, w2d, self.camera, self.cost_fun, pose_init=None, force_init_solve=True,
                               with_pose_opt_plus=True)[0]
            return pose
