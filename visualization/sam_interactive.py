import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import pycocotools.mask as mask_util
import cv2


def post_process_mask(mask, mode):

    mask = mask.squeeze()
    if mode == 'auto':
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.erode(mask, kernel, iterations=3)
    return mask


def plot_instructions(mode):
    if mode == 'auto':
        ax.set_title(f'Auto Mode: Left click to set instance points, right click to set background points, c to clear all points, space to save and exit')
    elif mode == 'manual':
        ax.set_title(f'Manual Mode: Click to record polygon vertex and press z to generate mask, c to clear all points, space to save and exit')

def auto_generate_mask(predictor, img, instance_points, background_points, mode):
    instance_points = np.array(instance_points)
    background_points = np.array(background_points)
    if background_points.shape[0] == 0:
        points = instance_points.astype(np.int32)
        label = np.ones((instance_points.shape[0])).astype(np.int32)
    else:
        points = np.concatenate((instance_points, background_points), axis=0).astype(np.int32)
        label = np.concatenate((np.ones((instance_points.shape[0])), np.zeros((background_points.shape[0]))),
                               axis=0).astype(np.int32)
    predictor.set_image(img)
    mask, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=label,
        multimask_output=False,
    )
    mask_bin = mask.astype(np.uint8).transpose(1, 2, 0)
    mask_bin = post_process_mask(mask_bin, mode)
    return mask_bin


def manual_generate_mask(points, dimensions, mode):
    mask = np.zeros((dimensions[0], dimensions[1]), dtype=np.uint8)
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    mask = mask // 255
    mask_bin = mask[np.newaxis, :, :]
    mask_bin = post_process_mask(mask_bin, mode)
    return mask_bin

def decode_mask(mask_coco, img):
    if len(mask_coco) == 1:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask = mask_util.decode(mask_coco)
    return mask


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


def filter_points(points):
    points = [x for x in points if x[0] and x[1] is not None]
    return points

def onclick(event):
    global instance_coords, bkgd_coords, mask, predictor, mode
    ix, iy = event.xdata, event.ydata

    if event.button == 1:  # Left click
        instance_coords.append([ix, iy])
        instance_coords = filter_points(instance_coords)
    elif event.button == 3:  # Right click
        bkgd_coords.append([ix, iy])
        bkgd_coords = filter_points(bkgd_coords)

    if mode == 'auto':
        if len(instance_coords) > 0:
            mask = auto_generate_mask(predictor, img, instance_coords, bkgd_coords, mode)

    ax.clear()
    # ax.plot(ix, iy, color)
    for x, y in instance_coords:
        ax.plot(x, y, 'bo')
    if len(bkgd_coords) > 0:
        for x, y in bkgd_coords:
            ax.plot(x, y, 'ro')
    if mode == 'auto':
        ax.imshow(img, alpha=0.5)  # Show the original image slightly transparent
        ax.imshow(mask, cmap='jet', alpha=0.5)  # Overlay the mask with transparency
        plot_instructions(mode)
    if mode == 'manual':
        ax.imshow(img)
        plot_instructions(mode)
    fig.canvas.draw()


def onkey(event):
    global instance_coords, bkgd_coords, mask, mode
    if event.key == 'c':
        instance_coords.clear()
        bkgd_coords.clear()
        ax.cla()
        plot_instructions(mode)
        ax.imshow(img)
        fig.canvas.draw()
    elif event.key == ' ':  # Check for space key
        plt.close(fig)  # Close the figure window
    elif event.key == 'z':
        if len(instance_coords) >= 3:
            mask = manual_generate_mask(instance_coords, img.shape[:2], mode)
            ax.imshow(img, alpha=0.5)  # Show the original image slightly transparent
            ax.imshow(mask, cmap='jet', alpha=0.5)  # Overlay the mask with transparency
            fig.canvas.draw()
    elif event.key == 'x':
        if mode == 'auto':
            mode = 'manual'
            instance_coords.clear()
            bkgd_coords.clear()
            ax.cla()
            plot_instructions(mode)
            ax.imshow(img)
            fig.canvas.draw()
        elif mode == 'manual':
            mode = 'auto'
            instance_coords.clear()
            bkgd_coords.clear()
            ax.cla()
            plot_instructions(mode)
            ax.imshow(img)
            fig.canvas.draw()



def interactive_mask(image, sam_predictor, init_mode='auto'):
    global instance_coords, bkgd_coords, ax, fig, img, mask, predictor, mode
    coords, mask = [], np.zeros(image.shape[:2], dtype=bool)
    img = image
    mode = init_mode
    predictor = sam_predictor
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img)
    plot_instructions(mode)
    instance_coords = []
    bkgd_coords = []
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

    return mask, mode


def plot_mask(img, masks):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    show_anns(masks, plot_bbox=False)
    plt.axis('off')
    plt.show()


def save_mask(mask_bin, crop_info):
    mask_coco = mask_util.encode(np.asfortranarray(mask_bin))#[0]
    mask_coco['counts'] = mask_coco['counts'].decode('utf-8')
    mask_coco.update({'crop_info': crop_info})
    mask_file = os.path.join(mask_dir, file_id.split('.')[0] + '.json')
    rle_json = json.dumps(mask_coco)
    with open(mask_file, "w") as file:
        file.write(rle_json)
    mask_vis = img * mask_bin[:, :, np.newaxis]
    mask_vis_file = os.path.join(mask_vis_dir, file_id.split('.')[0] + '.png')
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(mask_vis_file, mask_vis)


if __name__ == '__main__':
    root_dir = os.getcwd()
    init_mode = 'auto'
    data_dir = '../data/onepose_datasets/train_data'
    # data_dir = '../data/onepose_datasets/val_data'
    # data_dir = '../data/onepose_datasets/test_data'

    # sam_checkpoint = os.path.join(root_dir, 'checkpoint')
    sam_checkpoint = os.path.join(os.path.join(root_dir, 'checkpoint'), 'sam_vit_h_4b8939.pth')
    model_type = "vit_h"
    device = torch.device('cuda:1')
    # mode = 'binary_mask'
    mode = 'coco_rle'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    all_catagory = {}
    for obj_name in os.listdir(data_dir):
        obj_id = obj_name.split('-')[0]
        all_catagory[obj_id] = obj_name

    tar_obj_id_list = ['0410', '0413', '0414', '0415', '0416', '0418', '0420', '0421', '0443', '0445', '0448', '0460', '0461', '0462', '0463', '0464', '0465',
                       '0477', '0479', '0484', '0499', '0506', '0507', '0509', '0512', '0513', '0516', '0529', '0530', '0531', '0532', '0533', '0536', '0542',
                       '0545', '0546', '0549', '0556', '0561', '0562', '0563', '0566', '0567', '0569', '0571', '0572', '0573', '0574', '0575']
    tar_obj_id_list = ['0408', '0409', '0419', '0422', '0423', '0424', '0447', '0450', '0452', '0455', '0456', '0458',
                       '0459', '0466', '0468', '0469', '0470', '0471', '0472', '0473', '0474', '0476', '0480', '0483',
                       '0486', '0487', '0488', '0489', '0490', '0492', '0493', '0494', '0495', '0496', '0497', '0498',
                       '0500', '0501', '0502', '0503', '0504', '0508', '0510', '0511', '0517', '0518', '0519', '0520',
                       '0521', '0522', '0523', '0525', '0526', '0527', '0534', '0535', '0537', '0539', '0543', '0547',
                       '0548', '0550', '0551', '0552', '0557', '0558', '0559', '0560', '0564', '0565', '0568', '0570',
                       '0577', '0578', '0579', '0580', '0582', '0583', '0594', '0595']
    tar_obj_id_list = ['0573']
    # all_points = get_all_points(tar_obj_id_list, all_catagory)
    for tar_obj_id in tar_obj_id_list:
        tar_traj_list = ['1', '2', '3']
        tar_traj_list = ['4']
        # tar_traj_list = [str(len(os.listdir(os.path.join(data_dir, all_catagory[tar_obj_id]))) - 1)]
        # img_id = '0'
        for tar_traj in tar_traj_list:
            target_dir = os.path.join(data_dir, all_catagory[tar_obj_id])
            for traj in os.listdir(target_dir):
                if traj.endswith(tar_traj):
                    tar_traj = traj
                    break

            traj_dir = os.path.join(target_dir, tar_traj)
            rgb_dir = os.path.join(traj_dir, 'color')
            mask_dir = os.path.join(traj_dir, 'mask')
            os.makedirs(mask_dir, exist_ok=True)
            mask_vis_dir = os.path.join(traj_dir, 'mask_vis')
            os.makedirs(mask_vis_dir, exist_ok=True)
            total_length = len(os.listdir(rgb_dir))
            cur_length = 0
            valid_masks = os.listdir(mask_vis_dir)
            for file_id in os.listdir(rgb_dir):
                if int(file_id.split('.')[0]) % 5 != 0:
                    cur_length += 1
                    continue
                if file_id.split('.')[0] + '.png' in valid_masks:
                    continue
                rgb_file = os.path.join(rgb_dir, file_id)
                img = cv2.imread(rgb_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img, crop_info = pre_process_img(img)

                # prompt = all_points[tar_obj_id][tar_traj][file_id]
                # input_point = prompt['points']
                # input_label = prompt['label']
                # predictor.set_image(img)
                # mask, _, _ = predictor.predict(
                #     point_coords=input_point,
                #     point_labels=input_label,
                #     multimask_output=False,
                # )
                mask, init_mode = interactive_mask(img, predictor, init_mode=init_mode)
                save_mask(mask, crop_info)

                print('mask progress: {}/{}'.format(cur_length, total_length))
                # masks = mask_generator.generate(img)
                # mask = post_process_mask(masks, img)
                # if mask is None:
                #     mask_generator = init_sam(sam_checkpoint, model_type, device, mode=mode, retry=True)
                #     masks = mask_generator.generate(img)
                #     mask = post_process_mask(masks, img)
                # # plot_mask(img, [mask])
                # # save the binary mask
                # mask_file = os.path.join(mask_dir, file_id.split('.')[0] + '.json')
                # if mask is None:
                #     mask_coco = {}
                # else:
                #     mask_coco = mask['segmentation']
                # mask_coco.update({'crop_info': crop_info})
                # rle_json = json.dumps(mask_coco)
                # cur_length += 1
                # # print('mask progress: {}/{}'.format(cur_length, total_length))
                # with open(mask_file, "w") as file:
                #     file.write(rle_json)
                # mask_binary = decode_mask(mask_coco, img)
                # mask_vis = img * mask_binary[:, :, np.newaxis]
                # mask_vis_file = os.path.join(mask_vis_dir, file_id.split('.')[0] + '.png')
                # mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(mask_vis_file, mask_vis)
                # print('mask progress: {}/{}'.format(cur_length, total_length))



