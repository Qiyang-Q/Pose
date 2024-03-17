import torch.nn as nn
import torch
from src.models.epropnp.monte_carlo_pose_loss import MonteCarloPoseLoss
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F
import numpy as np
from src.utils import pose_utils


class PoseCriterion(nn.Module):
    def __init__(self, pose_scaling=1e-3, pose_epoch=30, pose_supervision=False, use_mc_loss=False):
        super(PoseCriterion, self).__init__()
        self.monte_carlo_pose_loss = MonteCarloPoseLoss()
        self.BCE_loss = nn.BCELoss(reduction='none')
        self.KL_div = nn.KLDivLoss(reduction='sum')
        self.softmax = nn.Softmax(dim=-1)
        self.pose_scaling = pose_scaling
        self.pose_epoch = pose_epoch
        self.use_mc_loss = use_mc_loss
        self.pose_supervision = pose_supervision
        self.alpha = 0.5
        self.gamma = 3.0

    def forward(self, pred, gt, epoch):
        pose_info = pred[0]
        pred_pose, logweights, cost_tgt = pose_info
        sim = pred[1]
        # sim = pose_utils.dual_softmax(sim)
        sim = F.softmax(sim, dim=-1)
        confidence = pred[2]
        scale = pred[3]
        gt_pose = gt['query_pose_quat']
        gt_sim = gt['gt_sim']

        # Pose Loss
        loss_r, loss_t, loss_mc = self.pose_loss(pred_pose, gt_pose, logweights, cost_tgt, scale=scale)
        loss_pose = (loss_r + loss_t + 0.2 * loss_mc) * self.pose_scaling

        # Similarity Loss
        loss_kl = self.KL_loss(sim, gt_sim)
        loss_mse = self.MSE_loss(sim, gt_sim)
        loss_focal = self.focal_loss(sim, gt_sim, gamma=self.gamma, alpha=self.alpha)
        loss_sim = loss_mse + loss_focal + loss_kl

        # Total Loss
        loss = loss_sim

        # End to End Pose Supervision works bad when the similarity matrix is not well trained,
        # so involve pose supervision after certain epoch
        if epoch >= self.pose_epoch and self.pose_supervision:
            loss = loss_sim + loss_pose

        # Logging
        log = self.loss_logging(loss, loss_r, loss_t, loss_mc, loss_sim, loss_kl, loss_mse, loss_focal, confidence)
        return loss, log


    def get_coords(self, res):
        x, y = np.meshgrid(range(res), range(res))
        coord = np.column_stack((y.ravel(), x.ravel()))
        return torch.from_numpy(coord).to(torch.float32)

    def loss_logging(self, loss, loss_r, loss_t, loss_mc, loss_sim, loss_kl, loss_mse, loss_focal, conf):
        return {'loss': loss.clone().detach().cpu().item(),
                'loss_r': loss_r.clone().detach().cpu().item(),
                'loss_t': loss_t.clone().detach().cpu().item(),
                'loss_mc': loss_mc.clone().detach().cpu().item(),
                'loss_sim': loss_sim.clone().detach().cpu().item(),
                'loss_kl': loss_kl.clone().detach().cpu().item(),
                'loss_mse': loss_mse.clone().detach().cpu().item(),
                'loss_focal': loss_focal.clone().detach().cpu().item(),
                'conf_max': conf.max().clone().detach().cpu().item(),}

    def MSE_loss(self, pred, gt):
        gt = gt[..., :-1]
        loss = (pred - gt).pow(2).mean()
        return loss

    def KL_loss(self, pred, gt):
        gt = gt[..., :-1]
        pred = self.softmax(pred)
        gt = self.softmax(gt)
        pred = torch.log(pred)
        loss = self.KL_div(pred, gt) / (pred.shape[0] * pred.shape[1])
        return loss

    def focal_loss(self, pred, gt, gamma=3.0, alpha=None, reduction='mean'):
        gt = gt[..., :-1]
        p = pred
        loss = self.BCE_loss(pred, gt)
        p_t = p * gt + (1 - p) * (1 - gt)
        loss = loss * ((1 - p_t) ** gamma)
        if alpha is not None:
            alpha_t = alpha * gt + (1 - alpha) * (1 - gt)
            loss = loss * alpha_t
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()

    def pose_loss(self, pred_pose, gt_pose, logweights=None, cost_tgt=None, scale=None):
        gt_pose = gt_pose.to(pred_pose.device)
        dot_quat = (pred_pose[:, None, 3:] @ gt_pose[:, 3:, None]).squeeze(-1).squeeze(-1)
        loss_t = (pred_pose[:, :3] - gt_pose[:, :3]).norm(dim=-1)
        beta = 0.05
        loss_t = torch.where(loss_t < beta, 0.5 * loss_t.square() / beta, loss_t - 0.5 * beta)
        loss_t = loss_t.mean()
        loss_r = (1 - dot_quat.square()) * 2
        loss_r = loss_r.mean()
        if self.use_mc_loss:
            # Monte Carlo Loss is used in EProPnP to imporve ambiguity problems, ok to ignore for now
            loss_mc = self.monte_carlo_pose_loss(logweights, cost_tgt, scale.detach().mean())
        else:
            loss_mc = torch.tensor(0.0)
        return loss_r, loss_t, loss_mc



