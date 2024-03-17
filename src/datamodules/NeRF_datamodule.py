from typing import Any

from lightning.pytorch import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from src.datasets.GATs_spg_dataset import GATsSPGDataset
from src.datasets.NeRF_dataset import NeRFDataset
from lightning.pytorch.utilities import move_data_to_device
from src.models.ibrnet.sample_ray import RaySamplerSingleImage


class NeRFDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.train_anno_file = kwargs['train_anno_file']
        self.val_anno_file = kwargs['val_anno_file']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.pin_memory = kwargs['pin_memory']
        self.num_leaf = kwargs['num_leaf']
        self.shape2d = kwargs['shape2d']
        self.shape3d = kwargs['shape3d']
        self.assign_pad_val = kwargs['assign_pad_val']
        self.num_source_views = kwargs['num_source_views']
        self.num_total_views = kwargs['num_total_views']
        self.resolution = kwargs['resolution']
        self.load_in_ram = kwargs['load_in_ram']
        self.dataset_dirs = kwargs['dataset_dirs']

        self.data_train = None
        self.data_val = None
        self.data_test = None

        # Loader parameters
        self.train_loader_params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """ Load data. Set variable: self.data_train, self.data_val, self.data_test"""
        trainset = NeRFDataset(
            anno_file=self.train_anno_file,
            num_leaf=self.num_leaf,
            split='train',
            shape2d=self.shape2d,
            shape3d=self.shape3d,
            pad_val=self.assign_pad_val,
            num_source_views=self.num_source_views,
            num_total_views=self.num_total_views,
            resolution=self.resolution,
            load_in_ram=self.load_in_ram,
            dataset_dirs=self.dataset_dirs + '/train_data'
            # dataset_dirs=self.dataset_dirs + '/test_data'
        )
        print("=> Read train anno file: ", self.train_anno_file)

        # valset = NeRFDataset(
        #     anno_file=self.val_anno_file,
        #     num_leaf=self.num_leaf,
        #     split='val',
        #     shape2d=self.shape2d,
        #     shape3d=self.shape3d,
        #     pad_val=self.assign_pad_val,
        #     load_pose_gt=True,
        #     num_source_views=self.num_source_views,
        #     num_total_views=self.num_total_views,
        #     resolution=self.resolution,
        #     load_in_ram=self.load_in_ram,
        #     dataset_dirs=self.dataset_dirs + '/val_data'
        # )
        # print("=> Read validation anno file: ", self.val_anno_file)

        self.data_train = trainset
        # self.data_val = valset
        # self.data_test = valset
    
    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, **self.train_loader_params)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.data_train, **self.train_loader_params)
    #
    # def test_dataloader(self):
    #     return DataLoader(dataset=self.data_test, **self.test_loader_params)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        ray_batch = RaySamplerSingleImage(batch)
        return ray_batch


