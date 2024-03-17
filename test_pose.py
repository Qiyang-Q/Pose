import os
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything, LightningModule, Trainer, Callback
import hydra
from omegaconf import DictConfig
from typing import List
from src.utils import template_utils as utils

import warnings

warnings.filterwarnings('ignore')


def test(config: DictConfig):
    if config['print_config']:
        utils.print_config(config)

    if "seed" in config:
        seed_everything(config['seed'])

    # Init PyTorch Lightning model ⚡
    model: LightningModule = hydra.utils.instantiate(config['model'])
    # model = LightningModule.load_from_checkpoint('data/models/checkpoints/onepose_plus_train/pose.ckpt')

    # Init PyTorch Lightning datamodule ⚡
    datamodule: LightningModule = hydra.utils.instantiate(config['datamodule'])
    datamodule.setup(stage='test')

    # Init PyTorch Lightning callbacks ⚡
    # callbacks: List[Callback] = []
    # if "callbacks" in config:
    #     for _, cb_conf in config['callbacks'].items():
    #         if "_target_" in cb_conf:
    #             callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init PyTorch Lightning loggers ⚡
    # if "logger" in config:
    #     for _, lg_conf in config['logger'].items():
    #         if "_target_" in lg_conf:
    #             logger = hydra.utils.instantiate(lg_conf)

    # Init PyTorch Lightning trainer ⚡
    trainer: Trainer = hydra.utils.instantiate(config['trainer'])

    # Evaluate model on test set after training
    trainer.test(model=model, datamodule=datamodule)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    return test(config)


if __name__ == "__main__":
    main()