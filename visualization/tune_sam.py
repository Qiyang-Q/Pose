import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import pycocotools.mask as mask_util


def decode_mask(mask_coco, img):
    if len(mask_coco) == 1:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask = mask_util.decode(mask_coco)
    return mask

def get_score(mask):
    center_points = np.array([[0.5, 0.5], [0.5, 0.6], [0.5, 0.4], [0.6, 0.5], [0.4, 0.5], [0.6, 0.6], [0.4, 0.4]])
    # offset = np.array([0.0, -0.1])
    # center_points = center_points + offset
    mask_bin = decode_mask(mask['segmentation'], img)
    mask_shape = mask_bin.shape
    center_points = (center_points * np.array([mask_shape[0], mask_shape[1]])).astype(np.int32)
    score = np.sum(mask_bin[center_points[:, 0], center_points[:, 1]])
    return score / len(center_points)

def get_coverage(mask):
    mask_area = mask['area']
    bbox_area = mask['bbox'][2] * mask['bbox'][3] + 1e-3
    return mask_area / bbox_area

def get_distance(mask):
    # bounding box is in xywh format
    middle_point = [mask['bbox'][0] + mask['bbox'][2] / 2, mask['bbox'][1] + mask['bbox'][3] / 2]
    dist = np.array(middle_point) - 256
    dist = np.sqrt(np.sum(dist ** 2, axis=0))
    return dist

def get_bbox_size(mask):
    bbox_area = mask['bbox'][2] * mask['bbox'][3]
    return bbox_area





def pre_process_img(img):
    # check row sum and column sum to remove black border
    # sum over all channels to find 0 for black pixels
    img_1d = np.sum(img, axis=-1)
    row_sum = np.sum(img_1d, axis=1)
    col_sum = np.sum(img_1d, axis=0)
    # find the 4 corners of the image
    row_idx = np.where(row_sum > 0)[0]
    col_idx = np.where(col_sum > 0)[0]
    if len(row_idx) == img.shape[0] and len(col_idx) == img.shape[1]:
        return img, None
    else:
        img = img[row_idx[0]:row_idx[-1]+1, col_idx[0]:col_idx[-1]+1]
        crop_info = [int(row_idx[0]), int(col_idx[0]), int(row_idx[-1]+1), int(col_idx[-1]+1)]
        return img, crop_info


def post_process_mask(masks, img):
    min_mask_size = int(img.shape[0] * img.shape[1] / 8)
    max_mask_size = int(img.shape[0] * img.shape[1]) * 0.8
    # filter out small masks with mask['area'] < min_mask_size
    mask = [m for m in masks if m['area'] > min_mask_size]
    # filter out large masks with mask['area'] > max_mask_size
    mask = [m for m in mask if m['area'] < max_mask_size]
    mask = [m for m in mask if get_bbox_size(m) < max_mask_size]
    mask = [m for m in mask if get_coverage(m) > 0.2]
    if len(mask) == 0:
        return None
    elif len(mask) > 1:
        # set masks threshold by 'crop_box' area / 'area', this should > 0.9
        # 'crop_box' has shape [x0, y0, x1, y1], get the area by (x1-x0)*(y1-y0)
        mask = [m for m in mask if get_score(m) > 0]
        if len(mask) == 0:
            return None
        mask = sorted(mask, key=lambda x: x['area'], reverse=True)
        # mask = sorted(mask, key=lambda x: x['area'] / (x['crop_box'][2] - x['crop_box'][0]) * (x['crop_box'][3] - x['crop_box'][1]), reverse=False)
        # mask = sorted(mask, key=lambda x: x['predicted_iou'] * x['stability_score'], reverse=True)
        return mask[0]
    return mask[0]
    # return mask


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_anns(anns, plot_bbox=False):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        if plot_bbox:
            # also draw bounding box according to ann['bbox']
            x0, y0 = ann['bbox'][0], ann['bbox'][1]
            w, h = ann['bbox'][2] - ann['bbox'][0], ann['bbox'][3] - ann['bbox'][1]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        img[m] = color_mask
    ax.imshow(img)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def plot_mask(img, masks):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    show_anns(masks, plot_bbox=False)
    plt.axis('off')
    plt.show()


def init_sam(checkpoint_dir, model_type, device, mode='binary_mask', retry=False):
    min_mask = int(512 * 512 / 10)

    sam_checkpoint = os.path.join(checkpoint_dir, 'sam_vit_h_4b8939.pth')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    if retry:
        mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                   points_per_batch=16*16,
                                                   pred_iou_thresh=0.4,
                                                   stability_score_thresh=0.4,
                                                   output_mode=mode,
                                                   min_mask_region_area=min_mask,)
        return mask_generator
    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                               points_per_batch=16*16,
                                               pred_iou_thresh=0.9,
                                               stability_score_thresh=0.9,
                                               output_mode=mode,
                                               min_mask_region_area=min_mask,)
    return mask_generator


if __name__ == '__main__':
    root_dir = os.getcwd()
    data_dir = '../data/onepose_datasets/val_data'
    sam_checkpoint = os.path.join(root_dir, 'checkpoint')
    model_type = "vit_h"
    device = torch.device('cuda:1')
    # mode = 'binary_mask'
    mode = 'coco_rle'
    mask_generator = init_sam(sam_checkpoint, model_type, device, mode=mode)
    all_catagory = {}
    for obj_name in os.listdir(data_dir):
        obj_id = obj_name.split('-')[0]
        all_catagory[obj_id] = obj_name

    tar_obj_id = '0620'
    tar_traj = '1'
    img_id_list = ['355']#, '745', '740', '745', '750']
    for img_id in img_id_list:
        img_id = img_id + '.png'
        target_dir = os.path.join(data_dir, all_catagory[tar_obj_id])
        for traj in os.listdir(target_dir):
            if traj.endswith(tar_traj):
                tar_traj = traj
                break

        traj_dir = os.path.join(target_dir, tar_traj)
        rgb_dir = os.path.join(traj_dir, 'color')
        intrinsic_dir = os.path.join(traj_dir, 'intrin_ba')
        pose_dir = os.path.join(traj_dir, 'poses_ba')
        bbox_dir = os.path.join(traj_dir, 'reproj_box')
        mask_dir = os.path.join(traj_dir, 'mask')
        os.makedirs(mask_dir, exist_ok=True)
        mask_vis_dir = os.path.join(traj_dir, 'mask_vis')
        os.makedirs(mask_vis_dir, exist_ok=True)
        total_length = len(os.listdir(rgb_dir))
        cur_length = 0

        rgb_file = os.path.join(rgb_dir, img_id)
        # intrinsic_file = os.path.join(intrinsic_dir, img_id + '.txt')
        # pose_file = os.path.join(pose_dir, img_id + '.txt')
        # bbox_file = os.path.join(bbox_dir, img_id + '.txt')

        img = cv2.imread(rgb_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, crop_info = pre_process_img(img)
        # K = np.loadtxt(intrinsic_file)
        # P = np.linalg.inv(np.loadtxt(pose_file))
        # B = np.loadtxt(bbox_file)

        masks = mask_generator.generate(img)
        mask = post_process_mask(masks, img)
        if mask is None:
            mask_generator = init_sam(sam_checkpoint, model_type, device, mode=mode, retry=True)
            masks = mask_generator.generate(img)
            mask = post_process_mask(masks, img)
        # plot_mask(img, [mask])
        # save the binary mask
        mask_file = os.path.join(mask_dir, img_id.split('.')[0] + '.json')
        if mask is None:
            mask_coco = {}
        else:
            mask_coco = mask['segmentation']
        mask_coco.update({'crop_info': crop_info})
        rle_json = json.dumps(mask_coco)
        cur_length += 1
        # print('mask progress: {}/{}'.format(cur_length, total_length))
        with open(mask_file, "w") as file:
            file.write(rle_json)
        mask_binary = decode_mask(mask_coco, img)
        mask_vis = img * mask_binary[:, :, np.newaxis]
        mask_vis_file = os.path.join(mask_vis_dir, img_id.split('.')[0] + '.png')
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_vis_file, mask_vis)
        print('mask progress: {}/{}'.format(cur_length, total_length))



