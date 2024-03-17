from typing import Any, Optional

import torch
import lightning.pytorch as pl

from itertools import chain

from lightning.pytorch.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS

from src.models.GATsSPG_architectures.GATs_SuperGlue import GATsSuperGlue
from src.losses.focal_loss import FocalLoss
from src.utils.eval_utils import compute_query_pose_errors, aggregate_metrics
from src.utils.vis_utils import draw_reprojection_pair
from src.utils.comm import gather
from src.models.extractors.SuperPoint.superpoint import SuperPoint
from src.sfm.extract_features import confs
from src.utils.model_io import load_network

from src.models.ibrnet.render_ray import render_rays
from src.models.ibrnet.render_image import render_single_image
from src.models.ibrnet.model import IBRNetModel
from src.models.ibrnet.mlp_network import IBRNet
from src.models.ibrnet.featnet_resfpn import ResNet
from src.models.ibrnet.featnet_resfpn import BasicBlock
from src.models.ibrnet.featnet_resfpn import Bottleneck
from src.models.ibrnet.sample_ray import RaySamplerSingleImage
from src.models.ibrnet.criterion import Criterion
from src.models.ibrnet.logging import log_scalars, log_img, LossLogger
from src.models.ibrnet.voxelize import Voxelizer
from src.models.epropnp.pnp_solver import pnp_solver
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


class LitModelNerf(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        pretrain_resnet = torch.load(self.hparams.resnet_path)
        self.feature_net = ResNet(BasicBlock, [3, 4, 6, 3], out_channel=self.hparams.feat_dim)
        pretrain_backbone = {k: v for k, v in pretrain_resnet.items() if 'bn' not in k and 'running' not in k}
        self.feature_net.load_state_dict(pretrain_backbone, strict=False)
        self.ibrnet = IBRNet(args, feat_dim=self.hparams.feat_dim, n_samples=self.hparams.N_samples)
        self.renderer = IBRNetModel(self.hparams, self.ibrnet, self.feature_net)
        self.crit = Criterion()
        self.loss_logger = LossLogger()
        self.log_img_step = self.hparams.log_img_step
        self.log_scalar_step = self.hparams.log_scalar_step

        # init training dataset again for voxelization

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        data_train = self.trainer.datamodule.data_train
        return data_train

    def forward(self, x):
        return self.renderer(x)

    def training_step(self, batch, batch_idx):
        ret = self.renderer(batch, self.local_rank)
        loss, log = self.crit(ret)
        self.loss_logger.log(log)
        return {'loss': loss, 'outputs': ret}

    def validation_step(self, batch, batch_idx):
        if self.local_rank == 0:
            log_img(self.renderer, self.logger, batch, self.global_step, mode='val')


    def test_step(self, batch, batch_idx):
        pass

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if self.local_rank == 0:
            if self.global_step % self.log_scalar_step == 0:
                self.loss_logger.write(self.logger, self.global_step)
            if self.global_step % self.log_img_step == 0:
                log_img(self.renderer, self.logger, batch, self.global_step, mode='train')

    def on_train_epoch_end(self):
        pass

    def teardown(self, stage: str) -> None:
        #extract voxels of objects trained on NeRF
        if self.local_rank == 0:
            self.renderer.to(f'cuda:{self.local_rank}')
            train_data = self.train_dataloader()
            voxelizer = Voxelizer(train_data, self.renderer, device=self.local_rank)
            voxelizer.voxelize()

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=self.hparams.milestones,
                                                                gamma=self.hparams.gamma)
            return [optimizer], [lr_scheduler]
        else:
            raise Exception("Invalid optimizer name.")