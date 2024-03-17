import json

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import json


def decode_mask(mask_coco):
    if 'counts' not in mask_coco.keys():
        return np.zeros((512, 512)).astype(np.uint8)
    if mask_coco['crop_info'] is not None:
        orig_res = 512
        mask = np.zeros((orig_res, orig_res)).astype(np.uint8)
        crop_info = mask_coco['crop_info']
        # crop_info = [row0, col0, row1, col1]
        mask[crop_info[0]:crop_info[2], crop_info[1]:crop_info[3]] = mask_util.decode(mask_coco)
        return mask
    mask = mask_util.decode(mask_coco)
    return mask


def load_mask(mask_dir):
    with open(mask_dir, 'r') as f:
        mask_coco = json.load(f)
    mask = decode_mask(mask_coco)
    return mask

def skew_symmetric(t):
    """ Convert a vector t to a skew-symmetric matrix """
    t = t.squeeze()
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])


def fundamental_matrix(K1, K2, R, T):
    Tx = skew_symmetric(T)
    # E = np.dot(K2.T, np.dot(Tx, R))  # Essential matrix
    # F = np.dot(np.linalg.inv(K2).T, np.dot(E, np.linalg.inv(K1)))  # Fundamental matrix
    E = np.linalg.solve(K2.T, R) @ Tx
    F = np.linalg.solve(K1.T, E.T).T
    return F


def draw_epipolar_line(img1, img2, F):
    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].axis('off')
    axs[0].set_title('Image 1')
    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].axis('off')
    axs[1].set_title('Image 2')
    plt.show(block=False)

    def onclick(event):
        if event.inaxes == axs[0]:
            x, y = int(event.xdata), int(event.ydata)

            # Mark the clicked point on image 1
            img1_with_point = img1.copy()
            radius_outer = 5  # Radius of the outer circle (white)
            cv2.circle(img1_with_point, (x, y), radius_outer, (255, 255, 255), -1)  # White circle

            # Draw a black circle (smaller radius) on top
            radius_inner = 3  # Radius of the inner circle (black)
            cv2.circle(img1_with_point, (x, y), radius_inner, (0, 0, 0), -1)
            # cv2.circle(img1_with_point, (x, y), 5, (255, 0, 0), -1)

            # Calculate the epipolar line on the second image
            pt = np.array([x, y, 1]).reshape(-1, 1)
            line = np.dot(F, pt)

            # Draw the epipolar line on the second image
            img2_with_line = img2.copy()
            height, width = img2.shape[:2]
            cv2.line(img2_with_line, (0, int(-line[2]/line[1])), (width, int(-(line[2]+line[0]*width)/line[1])), (255, 0, 0), 1)

            # Update the plots
            axs[0].imshow(cv2.cvtColor(img1_with_point, cv2.COLOR_BGR2RGB))
            axs[1].imshow(cv2.cvtColor(img2_with_line, cv2.COLOR_BGR2RGB))
            plt.draw()

    def on_key(event):
        if event.key == 'escape':
            plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)  # Pause to handle GUI events





if __name__ == '__main__':
    print(plt.get_backend())
    print(os.getcwd())
    data_dir = '../data/onepose_datasets/train_data'
    all_catagory = {}
    for obj_name in os.listdir(data_dir):
        obj_id = obj_name.split('-')[0]
        all_catagory[obj_id] = obj_name

    tar_obj_id = '0573'
    traj_pair = ['3', '4']
    img_pair = ['170', '415']

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

    img1 = cv2.imread(rgb_file1)
    img2 = cv2.imread(rgb_file2)
    # mask1 = load_mask(mask_file1)[..., np.newaxis]
    # mask2 = load_mask(mask_file2)[..., np.newaxis]
    # img1 = img1 * mask1
    # img2 = img2 * mask2

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

    R_rel = R2.T @ R1
    t_rel = R1.T @ (T2 - T1)
    F = fundamental_matrix(K1, K2, R_rel, t_rel)

    draw_epipolar_line(img1, img2, F)













