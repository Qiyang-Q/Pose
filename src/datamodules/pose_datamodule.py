import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from src.datasets.pose_dataset import PoseDataset
from src.models.pose.sample_data import DataSampler
from src.utils import data_utils


class PoseDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.train_anno_file = kwargs['train_anno_file']
        self.val_anno_file = kwargs['val_anno_file']
        self.test_anno_file = kwargs['test_anno_file']
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
        self.sample_res = kwargs['sample_resolution']
        self.load_in_ram = kwargs['load_in_ram']
        self.dataset_dirs = kwargs['dataset_dirs']
        self.voxel_samples = kwargs['voxel_samples']
        self.image_samples = kwargs['image_samples']
        self.crd_samples = kwargs['crd_samples']
        self.sigma = kwargs['sigma']
        self.query_dir = kwargs['query_dir'] if 'query_dir' in kwargs.keys() else None
        self.train_anno_file_unseen = kwargs['train_unseen_anno_file'] if 'train_unseen_anno_file' in kwargs.keys() else None

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
        if stage == 'fit' or stage is None:
            trainset = PoseDataset(
                anno_file=self.train_anno_file,
                num_leaf=self.num_leaf,
                split='train',
                shape2d=self.shape2d,
                shape3d=self.shape3d,
                pad_val=self.assign_pad_val,
                num_source_views=self.num_source_views,
                num_total_views=self.num_total_views,
                resolution=self.resolution,
                sample_res=self.sample_res,
                load_in_ram=self.load_in_ram,
                dataset_dirs=self.dataset_dirs + '/train_data'
            )
            print("=> Read train anno file: ", self.train_anno_file)
            self.data_train = trainset

        if stage == 'test':
            testset = PoseDataset(
                anno_file=self.train_anno_file,
                num_leaf=self.num_leaf,
                split='test',
                shape2d=self.shape2d,
                shape3d=self.shape3d,
                pad_val=self.assign_pad_val,
                load_pose_gt=True,
                num_source_views=self.num_source_views,
                num_total_views=self.num_total_views,
                resolution=self.resolution,
                sample_res=self.sample_res,
                load_in_ram=self.load_in_ram,
                dataset_dirs=self.dataset_dirs + '/train_data',
                query_dir=self.query_dir,
                test_anno=self.train_anno_file_unseen
            )
            print("=> Read test anno file: ", self.train_anno_file)
            self.data_test = testset
        # if stage == 'test':
        #     testset = PoseDataset(
        #         anno_file=self.test_anno_file,
        #         num_leaf=self.num_leaf,
        #         split='test',
        #         shape2d=self.shape2d,
        #         shape3d=self.shape3d,
        #         pad_val=self.assign_pad_val,
        #         load_pose_gt=True,
        #         num_source_views=self.num_source_views,
        #         num_total_views=self.num_total_views,
        #         resolution=self.resolution,
        #         sample_res=self.sample_res,
        #         load_in_ram=self.load_in_ram,
        #         dataset_dirs=self.dataset_dirs + '/test_data',
        #         query_dir=self.query_dir
        #     )
        #     print("=> Read test anno file: ", self.test_anno_file)
        #     self.data_test = testset

        # self.data_val = valset


    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, **self.train_loader_params, collate_fn=data_utils.batchify)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, **self.test_loader_params, collate_fn=data_utils.batchify)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        batch = DataSampler(batch, self.voxel_samples, self.image_samples, self.crd_samples, self.sigma)
        return batch


