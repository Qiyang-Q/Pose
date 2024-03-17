import numpy as np


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
        self.trans_loss.append(translation_distance)
        self.rot_loss.append(angular_distance)
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

    def evaluate(self, pose_pred, pose_gt):
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
