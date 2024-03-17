import numpy as np
from src.utils.pose_utils import quat2mat
import torch

class Evaluator():
    def __init__(self):
        self.cmd1 = []
        self.cmd3 = []
        self.cmd5 = []
        self.cmd7 = []
        self.cm1 = []
        self.cm3 = []
        self.cm5 = []
        self.d1 = []
        self.d3 = []
        self.d5 = []
        self.rot_loss = []
        self.trans_loss = []
        self.add = []
    
    def cm_degree_1_metric(self, pose_pred, pose_target):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_target[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cmd1.append(translation_distance < 1 and angular_distance < 1)
        self.cm1.append(translation_distance < 1)
        self.d1.append(angular_distance < 1)

    def cm_degree_5_metric(self, pose_pred, pose_target):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_target[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cmd5.append(translation_distance < 5 and angular_distance < 5)
        self.cm5.append(translation_distance < 5)
        self.d5.append(angular_distance < 5)

    def cm_degree_3_metric(self, pose_pred, pose_target):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_target[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cmd3.append(translation_distance < 3 and angular_distance < 3)
        self.cm3.append(translation_distance < 3)
        self.d3.append(angular_distance < 3)

    def pose_loss(self, pose_pred, pose_gt):
        dot_quat = (pose_pred[:, None, 3:] @ pose_gt[:, 3:, None]).squeeze(-1).squeeze(-1)
        loss_t = (pose_pred[:, :3] - pose_gt[:, :3]).norm(dim=-1)
        # beta = 0.05
        # loss_t = torch.where(loss_t < beta, 0.5 * loss_t.square() / beta, loss_t - 0.5 * beta)
        loss_t = loss_t.mean().cpu().numpy()
        loss_r = (1 - dot_quat.square()) * 2
        loss_r = loss_r.mean().cpu().numpy()
        angle_radians = 2 * torch.acos(dot_quat.clamp(min=-1, max=1))
        angle_degrees = (angle_radians * 180 / torch.pi).cpu().numpy()
        angle_degrees = min(angle_degrees, np.abs(360 - angle_degrees))
        self.rot_loss.append(angle_degrees)
        self.trans_loss.append(loss_t)
    
    def evaluate(self, pose_pred, pose_gt, pose_gt_quat):
        self.pose_loss(pose_pred, pose_gt_quat)
        if isinstance(pose_gt, torch.Tensor):
            if pose_gt.dim() == 3:
                pose_gt = pose_gt.squeeze(0)
            pose_gt = pose_gt.cpu().numpy()
        if pose_pred is None:
            self.cmd5.append(False)
            self.cmd1.append(False)
            self.cmd3.append(False)
            self.cmd7.append(False)
        else:
            if pose_pred.shape == (4, 4):
                pose_pred = pose_pred[:3, :4]
            if pose_gt.shape == (4, 4):
                pose_gt = pose_gt[:3, :4]
            if pose_pred.shape[-1] == 7:
                pose_pred = quat2mat(pose_pred).squeeze()
            self.cm_degree_1_metric(pose_pred, pose_gt)
            self.cm_degree_3_metric(pose_pred, pose_gt)
            self.cm_degree_5_metric(pose_pred, pose_gt)
    
    def summarize(self):
        cmd1 = np.mean(self.cmd1)
        cmd3 = np.mean(self.cmd3)
        cmd5 = np.mean(self.cmd5)
        cm1 = np.mean(self.cm1)
        cm3 = np.mean(self.cm3)
        cm5 = np.mean(self.cm5)
        d1 = np.mean(self.d1)
        d3 = np.mean(self.d3)
        d5 = np.mean(self.d5)
        rot = np.mean(self.rot_loss)
        trans = np.mean(self.trans_loss)
        print('1 cm 1 degree metric: {}'.format(cmd1))
        print('3 cm 3 degree metric: {}'.format(cmd3))
        print('5 cm 5 degree metric: {}'.format(cmd5))
        print(f'1 cm metric: {cm1}')
        print(f'3 cm metric: {cm3}')
        print(f'5 cm metric: {cm5}')
        print(f'1 deg metric: {d1}')
        print(f'3 deg metric: {d3}')
        print(f'5 deg metric: {d5}')
        print(f'rot_loss: {rot}')
        print(f'trans_loss: {trans}')

        self.cmd1 = []
        self.cmd3 = []
        self.cmd5 = []
        self.cmd7 = []
        return {'cmd1': cmd1, 'cmd3': cmd3, 'cmd5': cmd5}