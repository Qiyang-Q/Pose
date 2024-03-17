import json
import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import gc
from PIL import Image
import pycocotools.mask as mask_util
import json
from src.models.ibrnet.featnet_resfpn import ResNet
from src.models.ibrnet.featnet_resfpn import BasicBlock
from src.utils import data_utils
import torch.nn as nn
from src.dift.models.dift_sd import SDFeaturizer


class Demo:

    def __init__(self, imgs, ft, img_size):
        self.ft = ft  # NCHW
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = img_size

    def plot_img_pairs(self, fig_size=3, alpha=0.45, scatter_size=70):

        fig, axes = plt.subplots(1, self.num_imgs, figsize=(fig_size * self.num_imgs, fig_size))

        plt.tight_layout()

        for i in range(self.num_imgs):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('source image')
            else:
                axes[i].set_title('target image')

        num_channel = self.ft.size(1)
        cos = nn.CosineSimilarity(dim=1)

        def onclick(event):
            if event.inaxes == axes[0]:
                with torch.no_grad():

                    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

                    src_ft = self.ft[0].unsqueeze(0)
                    src_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(src_ft)
                    src_vec = src_ft[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1

                    del src_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    trg_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(self.ft[1:])
                    cos_map = cos(src_vec, trg_ft).cpu().numpy()  # N, H, W

                    del trg_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    axes[0].clear()
                    axes[0].imshow(self.imgs[0])
                    axes[0].axis('off')
                    axes[0].scatter(x, y, c='b', s=scatter_size)
                    axes[0].set_title('source image')

                    for i in range(1, self.num_imgs):
                        max_yx = np.unravel_index(cos_map[i - 1].argmax(), cos_map[i - 1].shape)
                        axes[i].clear()

                        heatmap = cos_map[i - 1]
                        heatmap = (heatmap - np.min(heatmap)) / (
                                    np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                        axes[i].imshow(self.imgs[i])
                        axes[i].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
                        axes[i].axis('off')
                        axes[i].scatter(max_yx[1].item(), max_yx[0].item(), c='b', s=scatter_size)
                        axes[i].set_title('target image')

                    del cos_map
                    del heatmap
                    gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

def fetch_state_dict(checkpoint, module_name):
    state_dict = {}
    for k, v in checkpoint.items():
        if k.split('.')[0] == module_name:
            k = '.'.join(k.split('.')[1:])
            state_dict[k] = v
    return state_dict

def dift_feature(dift, img):
    dift_feat = dift.forward(img,
                             prompt='',
                             t=261,
                             up_ft_index=2,
                             ensemble_size=8)
    return dift_feat.squeeze(0).cpu()



if __name__ == '__main__':
    res = 128
    dift = SDFeaturizer()
    device = 'cuda:1'
    pose_weight_path = '../data/models/checkpoints/onepose_plus_train/pose_mini.ckpt'
    nerf_mini_weight_path = '../data/models/checkpoints/onepose_plus_train/nerf_mini.ckpt'
    nerf_weight_path = '../data/models/checkpoints/onepose_plus_train/nerf_v3.ckpt'
    nerf_weight_path_v2 = '../data/models/checkpoints/onepose_plus_train/last.ckpt'
    weight = torch.load(nerf_weight_path_v2, map_location=device)['state_dict']
    state_dict = fetch_state_dict(weight, 'feature_net')
    feature_net = ResNet(BasicBlock, [3, 4, 6, 3], out_channel=128)
    feature_net.load_state_dict(state_dict)
    feature_net.to(device)
    feature_net.eval()

    print(plt.get_backend())
    print(os.getcwd())
    data_dir = '../data/onepose_datasets/test_data'
    all_catagory = {}
    for obj_name in os.listdir(data_dir):
        obj_id = obj_name.split('-')[0]
        all_catagory[obj_id] = obj_name

    tar_obj_id = '0459'
    traj_pair = ['1', '1']
    img_pair = ['180', '265']

    target_dir = os.path.join(data_dir, all_catagory[tar_obj_id])
    for traj in os.listdir(target_dir):
        if traj.endswith(traj_pair[0]):
            tar_traj1 = traj
        if traj.endswith(traj_pair[1]):
            tar_traj2 = traj


    traj_dir1 = os.path.join(target_dir, tar_traj1)
    traj_dir2 = os.path.join(target_dir, tar_traj2)
    rgb_dir1 = os.path.join(traj_dir1, 'color')
    rgb_dir2 = os.path.join(traj_dir2, 'color')
    intrinsic_dir1 = os.path.join(traj_dir1, 'intrin_ba')
    intrinsic_dir2 = os.path.join(traj_dir2, 'intrin_ba')
    pose_dir1 = os.path.join(traj_dir1, 'poses_ba')
    pose_dir2 = os.path.join(traj_dir2, 'poses_ba')
    mask_dir1 = os.path.join(traj_dir1, 'mask')
    mask_dir2 = os.path.join(traj_dir2, 'mask')

    rgb_file1 = os.path.join(rgb_dir1, img_pair[0] + '.png')
    rgb_file2 = os.path.join(rgb_dir2, img_pair[1] + '.png')

    intrinsic_file1 = os.path.join(intrinsic_dir1, img_pair[0] + '.txt')
    intrinsic_file2 = os.path.join(intrinsic_dir2, img_pair[1] + '.txt')

    pose_file1 = os.path.join(pose_dir1, img_pair[0] + '.txt')
    pose_file2 = os.path.join(pose_dir2, img_pair[1] + '.txt')

    mask_file1 = os.path.join(mask_dir1, img_pair[0] + '.json')
    mask_file2 = os.path.join(mask_dir2, img_pair[1] + '.json')

    img1, img_norm1, scaling = data_utils.load_rgb(rgb_file1, res=res)
    img2, img_norm2, scaling = data_utils.load_rgb(rgb_file2, res=res)
    img1_orig, _, _ = data_utils.load_rgb(rgb_file1, res=512)
    img2_orig, _, _ = data_utils.load_rgb(rgb_file2, res=512)
    with open(mask_file1, 'r') as f:
        mask1 = data_utils.load_mask(json.load(f))[..., np.newaxis]
    with open(mask_file2, 'r') as f:
        mask2 = data_utils.load_mask(json.load(f))[..., np.newaxis]
    img_norm1 = img_norm1 * mask1
    img_norm2 = img_norm2 * mask2

    K1 = np.loadtxt(intrinsic_file1)
    K2 = np.loadtxt(intrinsic_file2)

    P1 = np.linalg.inv(np.loadtxt(pose_file1))
    P2 = np.linalg.inv(np.loadtxt(pose_file2))
    # P1 = np.loadtxt(pose_file1)
    # P2 = np.loadtxt(pose_file2)
    R1 = P1[:3, :3]
    R2 = P2[:3, :3]
    T1 = P1[:3, 3:]
    T2 = P2[:3, 3:]


    with torch.no_grad():
        img_norm1_ = torch.from_numpy(img_norm1).permute(2, 0, 1).unsqueeze(0).to(device)
        img_norm2_ = torch.from_numpy(img_norm2).permute(2, 0, 1).unsqueeze(0).to(device)
        img1_ = torch.from_numpy(img1_orig).permute(2, 0, 1).unsqueeze(0).to(device)
        img2_ = torch.from_numpy(img2_orig).permute(2, 0, 1).unsqueeze(0).to(device)
        imgs = [img_norm1_, img_norm2_]
        ft = feature_net(torch.cat(imgs, dim=0))
        dift_ft1 = dift_feature(dift, img1_)
        dift_ft2 = dift_feature(dift, img2_)
        dift_ft = torch.stack([dift_ft1, dift_ft2], dim=0)
        imgs_vis = [img1, img2]
        demo = Demo(imgs_vis, ft.cpu(), 128)
        demo.plot_img_pairs()
        demo_dift = Demo(imgs_vis, dift_ft.cpu(), 128)
        demo_dift.plot_img_pairs()

        print('done')