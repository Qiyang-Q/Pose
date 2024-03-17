import gc
import matplotlib.pyplot as plt
import torch

import torch.nn as nn
import numpy as np
import imageio
import os
from PIL import Image
from src.utils.pose_utils import *
from src.dift.models.dift_sd import SDFeaturizer


class Demo:

    def __init__(self, imgs, ft, img_size, projs=None):
        self.ft = ft  # NCHW
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = img_size
        self.projs = projs
        self.scatter_size = 50

    def plot_img_pairs(self, fig_size=3, alpha=0.45, scatter_size=70):
        # if too many images, subplot to multiple rows
        col = 5
        # row = self.num_imgs // col
        row = (self.num_imgs + col - 1) // col
        fig, axes = plt.subplots(row, col, figsize=(20, row * 4))
        plt.tight_layout()

        for i in range(self.num_imgs):
            r = i // col
            c = i % col
            axes[r, c].imshow(self.imgs[i])
            axes[r, c].scatter(self.projs[i][1], self.projs[i][0], c='g', s=self.scatter_size)
            axes[r, c].axis('off')
            if i == 0:
                axes[r, c].set_title('source image')
            else:
                axes[r, c].set_title('target image')

        num_channel = self.ft.size(1)
        cos = nn.CosineSimilarity(dim=1)

        def onclick(event):
            if event.inaxes == axes[0, 0]:
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

                    axes[0, 0].clear()
                    axes[0, 0].imshow(self.imgs[0])
                    axes[0, 0].axis('off')
                    axes[0, 0].scatter(x, y, c='b', s=scatter_size)
                    axes[0, 0].set_title('source image')

                    for i in range(1, self.num_imgs):
                        r = i // col
                        c = i % col
                        max_yx = np.unravel_index(cos_map[i - 1].argmax(), cos_map[i - 1].shape)
                        axes[r, c].clear()

                        heatmap = cos_map[i - 1]
                        heatmap = (heatmap - np.min(heatmap)) / (
                                    np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                        axes[r, c].imshow(self.imgs[i])
                        axes[r, c].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
                        axes[r, c].axis('off')
                        axes[r, c].scatter(max_yx[1].item(), max_yx[0].item(), c='b', s=scatter_size)
                        axes[r, c].set_title('target image')

                    del cos_map
                    del heatmap
                    gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

def plot_img(img, featmap, ft_aggre, x2d_proj, fig_size=3, alpha=0.45, scatter_size=70):
    num_imgs = 1
    img_size = 128
    scale = img_size / featmap.size(2)
    fig, axes = plt.subplots(1, num_imgs, figsize=(fig_size * num_imgs, fig_size))
    featmap = featmap.unsqueeze(0)
    plt.tight_layout()

    # axes.imshow(img)
    axes.axis('off')
    # axes[1].axis('off')

    num_channel = featmap.size(1)
    cos = nn.CosineSimilarity(dim=1)

    with torch.no_grad():
        x2d_proj = torch.round(x2d_proj)
        x, y = int(x2d_proj[0].item()), int(x2d_proj[1].item())

        src_vec = ft_aggre.view(1, num_channel, 1, 1)
        # src_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(src_ft)
        # src_vec = src_ft[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1

        del ft_aggre
        gc.collect()
        torch.cuda.empty_cache()

        trg_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(featmap)
        cos_map = cos(src_vec, trg_ft).cpu().numpy()[0]  # N, H, W
        # dot_product_map = dot_product_softmax(src_vec, trg_ft)#.cpu().numpy()  # N, H, W

        del trg_ft
        gc.collect()
        torch.cuda.empty_cache()


        axes.clear()
        axes.imshow(img)
        axes.axis('off')
        axes.scatter(x, y, c='g', s=scatter_size)
        axes.set_title('source image')


        max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)

        heatmap = cos_map
        heatmap = (heatmap - np.min(heatmap)) / (
                    np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
        axes.imshow(255 * heatmap, alpha=alpha, cmap='viridis')
        # axes[0].axis('off')
        axes.scatter(max_yx[1].item(), max_yx[0].item(), c='b', s=scatter_size)



        del heatmap
        del cos_map
        # del dot_product_map
        gc.collect()
        # axes.imshow(img)
        axes.set_title('cosine_sim')
        # axes[1].set_title('dot_product')
    # fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def projection_2d(x3d, intrinsic, extrinsic):
    N = x3d.size(0)
    ones = torch.ones(N, 1, device=x3d.device, dtype=x3d.dtype)
    x3d_homogeneous = torch.cat([x3d, ones], dim=1)
    # Multiply with extrinsic
    points_cam = torch.mm(x3d_homogeneous, extrinsic.t())
    # Multiply with intrinsic
    x2d_homogeneous = torch.mm(points_cam, intrinsic.t())
    # Convert to cartesian coordinates
    x2d_proj = x2d_homogeneous[:, :2] / x2d_homogeneous[:, 2].unsqueeze(-1)
    # x2d_proj = x2d_proj.flip(-1)
    return x2d_proj#, x2d_homogeneous[:, 2]

def scatter_proj(x2d_proj, img, scatter_size=50):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(x2d_proj[:, 0], x2d_proj[:, 1], c='b', s=scatter_size)
    plt.show()


def vis_w2d(w2d, img, res):
    w2d_vis = nn.Upsample(size=(res, res), mode='bilinear')(w2d.permute(0, 3, 1, 2))
    w2d_vis = w2d_vis.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(w2d_vis[..., 0])
    plt.subplot(1, 3, 3)
    plt.imshow(w2d_vis[..., 1])
    plt.show()


def vis_aggre_feat(data, data_debug, w2d):

    img = data['rgb']
    intr = data['query_intr'][0]
    extr = data['query_pose'][0]
    ref_intr = data['ref_intr'][0]
    ref_extr = data['ref_pose'][0]
    ref_featmap = data_debug['ref_featmaps']
    ref_img = data['ref_rgbs'][0]
    voxel = data['voxel'][0]
    visibility = data['visibility'][0]
    inv_crd = data['inv_crd'][0]
    ref_x2d_proj = multiview_back_projection(voxel.unsqueeze(0), ref_intr.unsqueeze(0), ref_extr.unsqueeze(0)).permute(0, 2, 1, 3)[0]
    x2d_proj = projection_2d(voxel, intr, extr) * 2
    scatter_proj(x2d_proj.cpu().numpy(), img[0].cpu().numpy())
    vis_w2d(w2d, img[0].cpu().numpy(), 128)

    for i in range(voxel.shape[0]):
        crd = inv_crd[i][:, 0].to(torch.int64)
        vis = visibility[i].sum()
        img_query = img
        img_ref = ref_img[crd][:vis]
        feat_query = data_debug['featmap']
        feat_ref = ref_featmap[crd][:vis]
        imgs = torch.cat([img_query, img_ref], dim=0)
        imgs = imgs.cpu().numpy()
        ft = torch.cat([feat_query, feat_ref], dim=0).cpu()
        ref_proj = ref_x2d_proj[:, i][crd][:vis]
        proj =x2d_proj[i].unsqueeze(0).flip(-1)
        projs = torch.round(torch.cat([proj, ref_proj], dim=0)).cpu().numpy()
        # imgs = [img1_, img2_]
        # ft = feature_net(torch.cat(imgs, dim=0))
        # imgs_vis = [img1, img2]
        demo = Demo(imgs, ft, 128, projs)
        demo.plot_img_pairs()
        plot_img(img[0].cpu().numpy(), data_debug['featmap'][0], data_debug['feat_3d'][0, i], x2d_proj[i])
        print('done')

def show_img(img):
    plt.imshow(img)
    plt.show()

def vis_attn_feat(data, data_debug, encoder):
    layer_dict = data_debug['layer_dict']
    img = data['rgb']
    intr = data['query_intr'][0]
    extr = data['query_pose'][0]
    ref_intr = data['ref_intr'][0]
    ref_extr = data['ref_pose'][0]
    ref_featmap = data_debug['ref_featmaps']
    ref_img = data['ref_rgbs'][0]
    voxel = data['voxel'][0]
    visibility = data['visibility'][0]
    inv_crd = data['inv_crd'][0]
    ref_x2d_proj = multiview_back_projection(voxel.unsqueeze(0), ref_intr.unsqueeze(0), ref_extr.unsqueeze(0)).permute(0, 2, 1, 3)[0]
    x2d_proj = projection_2d(voxel, intr, extr) * 2
    scatter_proj(x2d_proj.cpu().numpy(), img[0].cpu().numpy())

    for i in range(voxel.shape[0]):
        crd = inv_crd[i][:, 0].to(torch.int64)
        vis = visibility[i].sum()
        img_query = img
        img_ref = ref_img[crd][:vis]
        feat_query = data_debug['featmap']
        feat_ref = ref_featmap[crd][:vis]
        imgs = torch.cat([img_query, img_ref], dim=0)
        imgs = imgs.cpu().numpy()
        ft = torch.cat([feat_query, feat_ref], dim=0).cpu()
        ref_proj = ref_x2d_proj[:, i][crd][:vis]
        proj =x2d_proj[i].unsqueeze(0).flip(-1)
        projs = torch.round(torch.cat([proj, ref_proj], dim=0)).cpu().numpy()
        # gt_attn = data['gt_sim'][0][i][:-1].reshape(64, 64).cpu().numpy()
        # show_img(gt_attn)
        # if gt_attn.sum() != 0:
        for layer in layer_dict.keys():
            feat_3d = layer_dict[layer][0][0][i]
            feat_2d = layer_dict[layer][1][0].reshape(64, 64, -1).permute(2, 0, 1)
            plot_img(img[0].cpu().numpy(), feat_2d, feat_3d, x2d_proj[i])
        print('done')
