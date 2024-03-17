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

import torch.nn as nn
from src.models.ibrnet.utils import img2mse
import torch.nn.functional as F
import torch

def get_loss_dict(loss, mse_loss, entropy_reg):
    loss = loss.detach().cpu().numpy()
    mse_loss = mse_loss.detach().cpu().numpy()
    entropy_reg = entropy_reg.detach().cpu().numpy()

    return {'loss': loss,
            'mse_loss': mse_loss,
            'entropy_reg': entropy_reg}


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, pose_pred):
        '''
        training criterion
        '''
        pred_rgb = outputs['rgb']
        pred_mask = outputs['mask'].float()
        gt_rgb = outputs['gt_rgb']
        weight = outputs['weights']
        entropy_reg = - torch.mean(weight * torch.log(weight + 1e-6))
        mse_loss = img2mse(pred_rgb, gt_rgb, pred_mask)
        loss = mse_loss + entropy_reg * 5e-2
        log = get_loss_dict(loss, mse_loss, entropy_reg)
        return loss, log
