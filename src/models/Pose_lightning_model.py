from typing import Any
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from src.models.ibrnet.model import IBRNetModel
from src.models.ibrnet.mlp_network import IBRNet
from src.models.ibrnet.featnet_resfpn import ResNet
from src.models.ibrnet.featnet_resfpn import BasicBlock
from src.models.ibrnet.criterion import Criterion
from src.models.pose.criterion import PoseCriterion
from src.models.ibrnet.logging import log_scalars, log_img, PoseLogger
from src.models.ibrnet.voxelize import Voxelizer
from src.models.epropnp.pnp_solver import pnp_solver
from src.models.matchers.match_network import Transformer
from src.models.pose.pose_model import pose_model
from src.evaluators import cmd_evaluator

def fetch_state_dict(checkpoint, module_name):
    state_dict = {}
    for k, v in checkpoint.items():
        if k.split('.')[0] == module_name:
            k = '.'.join(k.split('.')[1:])
            state_dict[k] = v
    return state_dict


class LitModelPose(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.test:
            self.eval()
        self.feature_net = ResNet(BasicBlock, [3, 4, 6, 3], out_channel=self.hparams.feat_dim)
        self.ibrnet = IBRNet(args, feat_dim=self.hparams.feat_dim, n_samples=self.hparams.N_samples)
        if self.hparams.freeze_backbone:
            self.freeze_backbone()
        if self.training:
            self.init_nerf_weights()
        self.renderer = IBRNetModel(self.hparams, self.ibrnet, self.feature_net)
        self.matcher = Transformer(d_model=self.hparams.head_dim,
                                   nhead=self.hparams.n_head,
                                   dustbin=self.hparams.dustbin,
                                   vis=self.hparams.vis_layer,
                                   num_encoder_layers=self.hparams.encoder_layers,
                                   num_decoder_layers=self.hparams.decoder_layers)

        self.pnp_solver = pnp_solver(self.training)
        self.pose_solver = pose_model(featnet=self.feature_net,
                                      ibrnet=self.ibrnet,
                                      matcher=self.matcher,
                                      pnp_solver=self.pnp_solver,
                                      use_nerf_mlp=self.hparams.use_nerf_mlp)
        self.render_crit = Criterion()
        self.pose_crit = PoseCriterion(pose_scaling=self.hparams.pose_scaling,
                                       pose_epoch=self.hparams.pose_epoch,
                                       pose_supervision=self.hparams.pose_supervision,
                                       use_mc_loss=self.hparams.use_mc_loss)
        if not self.training:
            self.load_checkpoint()
        self.log_img_step = self.hparams.log_img_step
        self.log_scalar_step = self.hparams.log_scalar_step
        self.pose_logging = PoseLogger()
        self.evaluator = cmd_evaluator.Evaluator()

    def init_nerf_weights(self):
        nerf_path = self.hparams.nerf_weight
        nerf_weight = torch.load(nerf_path, map_location=self.device)['state_dict']
        feat_weight = fetch_state_dict(nerf_weight, 'feature_net')
        ibr_weight = fetch_state_dict(nerf_weight, 'ibrnet')
        self.ibrnet.load_state_dict(ibr_weight, strict=True)
        self.feature_net.load_state_dict(feat_weight, strict=True)

    def load_checkpoint(self):
        checkpoint_path = self.hparams.checkpoint_dir
        checkpoint = torch.load(checkpoint_path, map_location=self.device)['state_dict']
        state_dict = fetch_state_dict(checkpoint, 'pose_solver')
        self.pose_solver.load_state_dict(state_dict, strict=False)

    def freeze_backbone(self):
        self.feature_net.eval()
        for param in self.feature_net.parameters():
            param.requires_grad = False
        self.ibrnet.eval()
        for param in self.ibrnet.parameters():
            param.requires_grad = False

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        data_train = self.trainer.datamodule.data_train
        return data_train

    def forward(self, x):
        return self.renderer(x)

    def training_step(self, batch, batch_idx):
        batch.update_status(self.local_rank)
        data = batch.sample_data()
        pred = self.pose_solver(data)
        loss, log = self.pose_crit(pred, data, self.current_epoch)
        self.pose_logging.log(log)
        return {'loss': loss, 'outputs': pred}

    def validation_step(self, batch, batch_idx):
        if self.local_rank == 0:
            log_img(self.renderer, self.logger, batch, self.global_step, mode='val')


    def test_step(self, batch, batch_idx):
        batch.update_status(self.local_rank, self.training)
        data = batch.sample_data()
        pose_pred = self.pose_solver(data)
        pose_gt = data['query_pose']
        self.evaluator.evaluate(pose_pred, pose_gt, data['query_pose_quat'])
        return None

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if self.local_rank == 0:
            if self.global_step % self.log_scalar_step == 0:
                self.pose_logging.write(self.logger, self.global_step)

    def on_test_end(self):
        self.evaluator.summarize()

    def on_validation_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        pass

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma)
        return [optimizer], [lr_scheduler]
