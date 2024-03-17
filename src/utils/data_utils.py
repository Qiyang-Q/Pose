import cv2
import torch
import numpy as np
import os.path as osp
import pycocotools.mask as mask_util
from loguru import logger
from pathlib import Path
import open3d as o3d
import imageio
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.spatial.transform import Rotation as R
# matplotlib.use('TkAgg')
import matplotlib.cm as cm
from torch.nn.utils.rnn import pad_sequence
rng = np.random.RandomState(234)
_EPS = np.finfo(float).eps * 4.0
TINY_NUMBER = 1e-8

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def pad_keypoints2d_random(keypoints, features, scores, img_h, img_w, n_target_kpts):
    dtype = keypoints.dtype
    
    n_pad = n_target_kpts - keypoints.shape[0]
    if n_pad < 0:
        keypoints = keypoints[:n_target_kpts] # [n_target_kpts, 2]
        features = features[:, :n_target_kpts] # [dim, n_target_kpts]
        scores = scores[:n_target_kpts] # [n_target_kpts, 1]
    else:
        while n_pad > 0:
            random_kpts_x = torch.randint(0, img_w, (n_pad, ), dtype=dtype)
            random_kpts_y = torch.randint(0, img_h, (n_pad, ), dtype=dtype)
            rand_kpts = torch.stack([random_kpts_y, random_kpts_x], dim=1)
            
            exist = (rand_kpts[:, None, :] == keypoints[None, :, :]).all(-1).any(1) # (n_pad, )
            kept_kpts = rand_kpts[~exist] # (n_kept, 2)
            n_pad -= len(kept_kpts)
            if len(kept_kpts) > 0:
                keypoints = torch.cat([keypoints, kept_kpts], 0)
                scores = torch.cat([scores, torch.zeros(len(kept_kpts), 1, dtype=scores.dtype)], dim=0)
                features = torch.cat([features, torch.ones(features.shape[0], len(kept_kpts))], dim=1)
   
    return keypoints, features, scores


def pad_features(features, num_leaf):
    num_features = features.shape[0]
    feature_dim = features.shape[1]
    n_pad = num_leaf - num_features

    if n_pad <= 0:
        features = features[:num_leaf]
    else:
        features = torch.cat([features, torch.ones((num_leaf - num_features, feature_dim))], dim=0)
    
    return features.T


def pad_scores(scores, num_leaf):
    num_scores = scores.shape[0]
    n_pad = num_leaf - num_scores

    if n_pad <= 0:
        scores = scores[:num_leaf]
    else:
        scores = torch.cat([scores, torch.zeros((num_leaf - num_scores, 1))], dim=0)

    return scores


def avg_features(features):
    ret_features = torch.mean(features, dim=0).reshape(-1, 1)
    return ret_features


def avg_scores(scores):
    ret_scores = torch.mean(scores, dim=0).reshape(-1, 1)
    return ret_scores


def pad_keypoints3d_random(keypoints, n_target_kpts):
    """ Pad or truncate orig 3d keypoints to fixed size."""
    n_pad = n_target_kpts - keypoints.shape[0]
    
    if n_pad < 0:
        keypoints = keypoints[:n_target_kpts] # [n_target_kpts: 3] 
    else :
        while n_pad > 0:
            rand_kpts_x = torch.rand(n_pad, 1) - 0.5 # zero mean
            rand_kpts_y = torch.rand(n_pad, 1) - 0.5 # zero mean
            rand_kpts_z = torch.rand(n_pad, 1) - 0.5 # zero mean
            rand_kpts = torch.cat([rand_kpts_x, rand_kpts_y, rand_kpts_z], dim=1) # [n_pad, 3]

            exist = (rand_kpts[:, None, :] == keypoints[None, :, :]).all(-1).any(1)
            kept_kpts = rand_kpts[~exist] # [n_kept, 3]
            n_pad -= len(kept_kpts)

            if len(kept_kpts) > 0:
                keypoints = torch.cat([keypoints, kept_kpts], dim=0)

    return keypoints


def pad_features3d_random(descriptors, scores, n_target_shape):
    """ Pad or truncate orig 3d feature(descriptors and scores) to fixed size."""
    dim = descriptors.shape[0]
    n_pad = n_target_shape - descriptors.shape[1]

    if not isinstance(descriptors, torch.Tensor):
        descriptors = torch.Tensor(descriptors)
    if not isinstance(scores, torch.Tensor):
        scores = torch.Tensor(scores)

    if n_pad < 0:
        descriptors = descriptors[:, :n_target_shape]
        scores = scores[:n_target_shape, :]
    else:
        descriptors = torch.cat([descriptors, torch.ones(dim, n_pad)], dim=-1)
        scores = torch.cat([scores, torch.zeros(n_pad, 1)], dim=0)
    
    return descriptors, scores


def build_features3d_leaves(descriptors, scores, idxs, n_target_shape, num_leaf):
    """ Given num_leaf, fix the numf of 3d features to n_target_shape * num_leaf""" 
    if not isinstance(descriptors, torch.Tensor):
        descriptors = torch.Tensor(descriptors)
    if not isinstance(scores, torch.Tensor):
        scores = torch.Tensor(scores)

    dim = descriptors.shape[0]
    orig_num = idxs.shape[0]
    n_pad = n_target_shape - orig_num

    # pad dustbin descriptors and scores
    descriptors_dustbin = torch.cat([descriptors, torch.ones(dim, 1)], dim=1)
    scores_dustbin = torch.cat([scores, torch.zeros(1, 1)], dim=0)
    dustbin_id = descriptors_dustbin.shape[1] - 1
    
    upper_idxs = np.cumsum(idxs, axis=0)
    lower_idxs = np.insert(upper_idxs[:-1], 0, 0)
    affilicate_idxs_ = []
    for start, end in zip(lower_idxs, upper_idxs):
        if num_leaf > end - start:
            idxs = np.arange(start, end).tolist()
            idxs += [dustbin_id] * (num_leaf - (end - start))
            shuffle_idxs = np.random.permutation(np.array(idxs)) 
            affilicate_idxs_.append(shuffle_idxs)
        else:
            shuffle_idxs = np.random.permutation(np.arange(start, end))[:num_leaf]
            affilicate_idxs_.append(shuffle_idxs)
         
    affilicate_idxs = np.concatenate(affilicate_idxs_, axis=0)

    assert affilicate_idxs.shape[0] == orig_num * num_leaf
    descriptors = descriptors_dustbin[:, affilicate_idxs] # [dim, num_leaf * orig_num]
    scores = scores_dustbin[affilicate_idxs, :] # [num_leaf * orig_num, 1]
    
    if n_pad < 0:
        descriptors = descriptors[:, :num_leaf * n_target_shape]
        scores = scores[:num_leaf * n_target_shape, :] 
    else:
        descriptors = torch.cat([descriptors, torch.ones(dim, n_pad * num_leaf)], dim=-1)
        scores = torch.cat([scores, torch.zeros(n_pad * num_leaf, 1)], dim=0)

    return descriptors, scores
    

def reshape_assign_matrix(assign_matrix, orig_shape2d, orig_shape3d, 
                          shape2d, shape3d, pad=True, pad_val=0):
    """ Reshape assign matrix (from 2xk to nxm)"""
    assign_matrix = assign_matrix.long()
    
    if pad:
        conf_matrix = torch.zeros(shape2d, shape3d, dtype=torch.int16)
        
        valid = (assign_matrix[0] < shape2d) & (assign_matrix[1] < shape3d)
        assign_matrix = assign_matrix[:, valid]

        conf_matrix[assign_matrix[0], assign_matrix[1]] = 1
        conf_matrix[orig_shape2d:] = pad_val
        conf_matrix[:, orig_shape3d:] = pad_val
    else:
        conf_matrix = torch.zeros(orig_shape2d, orig_shape3d, dtype=torch.int16)
        
        valid = (assign_matrix[0] < shape2d) & (assign_matrix[1] < shape3d)
        conf_matrix = conf_matrix[:, valid]
        
        conf_matrix[assign_matrix[0], assign_matrix[1]] = 1
    
    return conf_matrix


def get_image_crop_resize(image, box, resize_shape):
    """Crop image according to the box, and resize the cropped image to resize_shape
    @param image: the image waiting to be cropped
    @param box: [x0, y0, x1, y1]
    @param resize_shape: [h, w]
    """
    center = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
    scale = np.array([box[2] - box[0], box[3] - box[1]])
    
    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    image_crop = cv2.warpAffine(image, trans_crop, (resize_w, resize_h), flags=cv2.INTER_LINEAR)

    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)
    return image_crop, trans_crop_homo


def get_K_crop_resize(box, K_orig, resize_shape):
    """Update K (crop an image according to the box, and resize the cropped image to resize_shape) 
    @param box: [x0, y0, x1, y1]
    @param K_orig: [3, 3] or [3, 4]
    @resize_shape: [h, w]
    """
    center = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
    scale = np.array([box[2] - box[0], box[3] - box[1]]) # w, h
    
    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)

    if K_orig.shape == (3, 3):
        K_orig_homo = np.concatenate([K_orig, np.zeros((3, 1))], axis=-1)
    else:
        K_orig_homo = K_orig.copy()
    assert K_orig_homo.shape == (3, 4)

    K_crop_homo = trans_crop_homo @ K_orig_homo # [3, 4]
    K_crop = K_crop_homo[:3, :3]
    
    return K_crop, K_crop_homo


def read_gray_scale(img_file):
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    image = image[None]

    return image

def get_K(intrin_file):
    assert Path(intrin_file).exists()
    with open(intrin_file, 'r') as f:
        lines = f.readlines()
    intrin_data = [line.rstrip('\n').split(':')[1] for line in lines]
    fx, fy, cx, cy = list(map(float, intrin_data))

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    K_homo = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])
    return K, K_homo

def video2img(video_path, outdir, downsample=1):
    Path(outdir).mkdir(exist_ok=True, parents=True)
    cap = cv2.VideoCapture(video_path)
    index = 0

    logger.info('Begin parsing video...')
    while True:
        ret, image = cap.read()
        if not ret:
            break
        
        if index % downsample == 0:
            image_path = osp.join(outdir, '{}.png'.format(index // downsample))
            cv2.imwrite(image_path, image)
        index += 1
    logger.info('Finish parsing video, images output to {outdir}')

def imgnet_norm(arr):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    arr = arr.astype(np.float32)
    mean_orig = np.mean(arr, axis=(1, 2), keepdims=True)
    std_orig = np.std(arr, axis=(1, 2), keepdims=True)

    arr -= mean_orig
    arr /= (std_orig + 1e-7)  # Add a small number to avoid division by zero
    arr *= std
    arr += mean

    return arr


def obj_root_dir(file_path):
    path_obj = Path(file_path)
    root_path = path_obj.parent.parent
    return str(root_path)

def parent_dir(file_path):
    path_obj = Path(file_path)
    parent_path = path_obj.parent
    return str(parent_path)


def obj2cam(pose):
    return np.linalg.inv(pose).astype(np.float32)


def load_multiview_cam_poses(data):
    cam_poses = []
    for data_id in data.keys():
        cam_poses.append(data[data_id]['cam_pose'])
    return np.stack(cam_poses)


def load_multiview_obj_poses(data):
    obj_poses = []
    for data_id in data.keys():
        obj_poses.append(data[data_id]['obj_pose'])
    return np.stack(obj_poses)


def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))


def batched_angular_dist_rot_matrix_2d(R1, R2):
    R1 = np.arctan2(R1[:, 1, 0], R1[:, 0, 0])
    R2 = np.arctan2(R2[:, 1, 0], R2[:, 0, 0])
    angular_dist = np.abs(R1 - R2)
    angular_dist = np.minimum(angular_dist, 2 * np.pi - angular_dist)
    return angular_dist


def skew_symmetric(t):
    """ Convert a vector t to a skew-symmetric matrix """
    t = t.squeeze()
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])


def sample_views(num_samples=12):
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / num_samples)
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    # xy_poses = np.zeros((num_samples, 3, 3))
    # xy_poses[:, :2, :2] = rotation_matrix.transpose(2, 0, 1)
    # xy_poses[:, 2, 2] = 1
    return rotation_matrix.transpose(2, 0, 1)


def plot_rotation_matrix_2d(rot_matrices):
    # Original vector (unit vector along x-axis)
    original_vector = np.array([1, 0])

    # Number of rotation matrices
    n = rot_matrices.shape[0]

    # Generate a color map to distinguish different rotations
    colors = cm.rainbow(np.linspace(0, 1, n))

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.quiver(0, 0, original_vector[0], original_vector[1], angles='xy', scale_units='xy', scale=1, color='blue',
               label='Original Vector')

    for i, rotation_matrix in enumerate(rot_matrices):
        # Apply the rotation matrix
        rotated_vector = np.dot(rotation_matrix, original_vector)

        # Plot the rotated vector
        plt.quiver(0, 0, rotated_vector[0], rotated_vector[1], angles='xy', scale_units='xy', scale=1, color=colors[i],
                   alpha=0.7)

    # Setting plot limits and labels
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.grid()
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.gca().set_aspect('equal', adjustable='box')

    # Show plot
    plt.legend()
    plt.show()


def sampled_view_density(all_poses, angular_thresh=np.pi/18):
    num_samples = int(np.pi / angular_thresh)
    sampled_poses = sample_views(num_samples=num_samples)
    # plot_rotation_matrix_2d(sampled_poses[:, :2, :2])
    xz_poses = np.zeros((len(all_poses), 2, 2))
    xz_poses[:, 0, 0] = all_poses[:, 0, 0]
    xz_poses[:, 0, 1] = all_poses[:, 0, 2]
    xz_poses[:, 1, 0] = all_poses[:, 2, 0]
    xz_poses[:, 1, 1] = all_poses[:, 2, 2]
    cam_pose_indices = np.arange(len(all_poses))
    ind_dict = {i: [] for i in cam_pose_indices}
    for i in range(len(sampled_poses)):
        batched_tar_pose = sampled_poses[i][None, ...].repeat(len(all_poses), 0)
        angular_dists = batched_angular_dist_rot_matrix_2d(batched_tar_pose, xz_poses)
        nearby_ind = cam_pose_indices[angular_dists <= angular_thresh]
        for ind in nearby_ind:
            ind_dict[ind].append(len(nearby_ind))
        if len(nearby_ind) == 0:
            sort_ind = np.argsort(angular_dists)
            for j in range(2):
                # ind_dict[sort_ind[j]].append((angular_thresh / angular_dists[sort_ind[j]]) ** 10)
                ind_dict[sort_ind[j]].append(1e-12)

    ind_dict = {i: np.min(ind_dict[i]) for i in ind_dict}
    mean_density = np.mean([ind_dict[i] for i in ind_dict])
    return mean_density, ind_dict


def get_nearest_pose_ids(tar_pose, ref_poses, num_select, tar_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams-1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    ids = np.arange(num_cams)
    # sorted_ids = np.argsort(dists)

    # filter out views with too large or too small angular distance
    # filter_condition = (dists <= np.pi / 9) & (dists >= np.pi / 36)
    filter_condition = (dists <= np.pi / 6)

    selected_ids = ids[filter_condition]
    if len(selected_ids) < num_select:
        selected_ids = np.argsort(dists)[:num_select]
    # selected_ids = ids#[:num_select]
    assert tar_id not in selected_ids

    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids


def load_rgb(dir, res=128):
    tgt_res = (res, res)
    img_orig = Image.open(dir)
    rgb = np.array(img_orig.resize(tgt_res, Image.BICUBIC)).astype(np.float32) / 255.
    scaling = res / img_orig.height
    return rgb, imgnet_norm(rgb[None, ...])[0], scaling


def load_src_rgbs(catagory, ids, res=128):
    tgt_res = (res, res)
    src_paths = [catagory[id]['img_file'] for id in ids]
    rgb = [np.array(Image.open(src_paths[i]).resize(tgt_res, Image.BICUBIC)) for i in range(len(src_paths))]
    rgb = np.array(rgb).astype(np.float32) / 255.
    return imgnet_norm(rgb)


def load_mask(mask, res=128):
    tgt_res = (res, res)
    render_mask = decode_mask(mask)
    render_mask = np.array(Image.fromarray(render_mask).resize(tgt_res, Image.NEAREST))
    return render_mask


def decode_mask(mask_coco):
    if mask_coco['crop_info'] is not None:
        orig_res = 512
        mask = np.zeros((orig_res, orig_res)).astype(np.uint8)
        crop_info = mask_coco['crop_info']
        # crop_info = [row0, col0, row1, col1]
        mask[crop_info[0]:crop_info[2], crop_info[1]:crop_info[3]] = mask_util.decode(mask_coco)
    else:
        mask = mask_util.decode(mask_coco)
    return mask


def rectify_intrinsic(intrinsic, scaling):
    is_batched = intrinsic.dim() == 3
    dtype = intrinsic.dtype
    intrinsic[..., :2, :] *= scaling
    if is_batched:
        B = intrinsic.size(0)
        extended = torch.eye(4, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    else:
        extended = torch.eye(4, dtype=dtype)
    extended[..., :3, :3] = intrinsic
    return extended


def load_intrinsics(catagory_anno, render_id, src_id, scaling):
    render_intrin = catagory_anno[render_id]['intrinsic']
    src_intrin = [catagory_anno[id]['intrinsic'] for id in src_id]
    render_intrin = torch.from_numpy(render_intrin).to(torch.float32)
    src_intrin = torch.from_numpy(np.array(src_intrin)).to(torch.float32)
    render_intrin = rectify_intrinsic(render_intrin, scaling)
    src_intrin = rectify_intrinsic(src_intrin, scaling)
    return render_intrin, src_intrin


def get_depth_range(pose, radius):
    # origin_depth = np.linalg.norm(pose[:3, 3])
    scaling = 1.5
    radius = radius * scaling
    origin_depth = pose[2, 3]
    return torch.tensor([origin_depth - radius, origin_depth + radius]).to(torch.float32)


def get_radius(bbox_file):
    bbox3d = np.loadtxt(bbox_file)
    x_range = bbox3d[:, 0].max() - bbox3d[:, 0].min()
    y_range = bbox3d[:, 1].max() - bbox3d[:, 1].min()
    z_range = bbox3d[:, 2].max() - bbox3d[:, 2].min()
    radius = max(x_range, y_range, z_range) / 2
    return radius


def norm_coor_3d(coor):
    coor_x, coor_y, coor_z = coor[..., 0], coor[..., 1], coor[..., 2]
    min_x, max_x = torch.min(coor_x, dim=-1, keepdim=True)[0], torch.max(coor_x, dim=-1, keepdim=True)[0]
    min_y, max_y = torch.min(coor_y, dim=-1, keepdim=True)[0], torch.max(coor_y, dim=-1, keepdim=True)[0]
    min_z, max_z = torch.min(coor_z, dim=-1, keepdim=True)[0], torch.max(coor_z, dim=-1, keepdim=True)[0]
    coor_x = (coor_x - min_x) / (max_x - min_x)
    coor_y = (coor_y - min_y) / (max_y - min_y)
    coor_z = (coor_z - min_z) / (max_z - min_z)
    # normalize to [-1, 1]
    coor_x = coor_x * 2 - 1
    coor_y = coor_y * 2 - 1
    coor_z = coor_z * 2 - 1
    return torch.stack((coor_x, coor_y, coor_z), dim=-1)


def get_inverse_crd(crd):
    inv_crd = {}
    for img_id in crd.keys():
        voxel_idx = crd[img_id][:, -1].astype(np.int32)
        transparency = crd[img_id][:, -2].astype(np.float32)
        for i in range(len(voxel_idx)):
            idx = voxel_idx[i]
            t = transparency[i]
            if idx not in inv_crd.keys():
                inv_crd[idx] = [[], []]
                inv_crd[idx][0].append(img_id)
                inv_crd[idx][1].append(t)
            else:
                inv_crd[idx][0].append(img_id)
                inv_crd[idx][1].append(t)
    for i in inv_crd.keys():
        inv_crd[i][0] = torch.tensor(inv_crd[i][0])
        inv_crd[i][1] = torch.tensor(inv_crd[i][1])
    # sort the key in ascending order
    inv_crd = {k: inv_crd[k] for k in sorted(inv_crd.keys())}
    return inv_crd


def get_crd_sampling_prob(inv_crd, sampling_threshold=40):
    sampling_prob = []
    for i in inv_crd.keys():
        if len(inv_crd[i]) < sampling_threshold:
            sampling_prob.append(0)
        else:
            sampling_prob.append(1)
    sampling_prob = torch.tensor(sampling_prob) * (1 / sum(sampling_prob))
    return sampling_prob


def sample_crd(x3d, x3d_norm, inv_crd, correspondence, crd_threshold=40, sample=True):
    if sample:
        sampling_idx = []
        keys = list(inv_crd.keys())
        for i in keys:
            if inv_crd[i][0].shape[0] > crd_threshold:
                sampling_idx.append(i)
            else:
                del inv_crd[i]
        old_idx = sampling_idx
        new_idx = np.arange(len(sampling_idx))
        mapping = dict(zip(old_idx, new_idx))
        # inv_crd = {mapping[key]: value for key, value in inv_crd.items()}

        sampling_idx = np.array(sampling_idx)
        x3d = x3d[sampling_idx]
        x3d_norm = x3d_norm[sampling_idx]
        sampling_idx = sampling_idx.astype(np.int16)
        for i in correspondence.keys():
            img_crd = correspondence[i][:, -1].astype(np.int16)
            match_idx = np.where(np.isin(img_crd, sampling_idx))[0]
            correspondence[i] = correspondence[i][match_idx]
            correspondence[i][:, -1] = np.vectorize(mapping.get)(correspondence[i][:, -1])
    inv_crd = list(inv_crd.values())
    for i, crd in enumerate(inv_crd):
        inv_crd[i] = torch.stack(crd, dim=-1)
    return x3d, x3d_norm, inv_crd, correspondence


def get_coords(tgt_res, scaling=1):
    tgt_res = int(tgt_res * scaling)
    x, y = np.meshgrid(range(tgt_res), range(tgt_res))
    coord = np.column_stack((y.ravel(), x.ravel()))
    return torch.from_numpy(coord).to(torch.float32)


def scalar_first_quat(q):
    return np.array([q[3], q[0], q[1], q[2]])


def mtx2quat(mtx):
    rot = mtx[:3, :3]
    tra = mtx[:3, 3]
    r_obj = R.from_matrix(rot)
    quat = np.hstack([tra.squeeze(), scalar_first_quat(r_obj.as_quat())])
    return quat


def vis_voxel(voxel):
    if isinstance(voxel, torch.Tensor):
        voxel = voxel.numpy()
    pcd_voxel = o3d.geometry.PointCloud()
    pcd_voxel.points = o3d.utility.Vector3dVector(voxel)
    pcd_voxel.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd_voxel])


def batchify(batch):
    output = {}
    output['rgb'] = torch.stack([b['rgb'] for b in batch], dim=0)
    output['query_id'] = [b['query_id'] for b in batch]
    output['pixel'] = torch.stack([b['pixel'] for b in batch], dim=0)
    output['rgb_norm'] = torch.stack([b['rgb_norm'] for b in batch], dim=0)
    output['ref_rgbs'] = [b['ref_rgbs'] for b in batch]
    # output['mask'] = torch.stack([b['mask'] for b in batch], dim=0)
    output['query_cam'] = torch.stack([b['query_cam'] for b in batch], dim=0)
    output['query_pose'] = torch.stack([b['query_pose'] for b in batch], dim=0)
    output['query_pose_quat'] = torch.stack([b['query_pose_quat'] for b in batch], dim=0)
    output['ref_pose'] = [b['ref_pose'] for b in batch]
    output['ref_trans'] = (pad_sequence([p[:, :3, 3] for p in output['ref_pose']], batch_first=True), torch.tensor([len(b['ref_pose']) for b in batch]))
    output['query_intr'] = torch.stack([b['query_intr'] for b in batch], dim=0)
    output['ref_intr'] = [b['ref_intr'] for b in batch]
    output['voxel'] = (pad_sequence([b['voxel'] for b in batch], batch_first=True), torch.tensor([len(b['voxel']) for b in batch]))
    output['voxel_norm'] = pad_sequence([b['voxel_norm'] for b in batch], batch_first=True)
    if batch[0]['crd'] != None:
        output['crd'] = (pad_sequence([b['crd'] for b in batch], batch_first=True, padding_value=-1), torch.tensor([len(b['crd']) for b in batch]))
    output['inv_crd'] = [b['inv_crd'] for b in batch]
    output['depth_range'] = torch.stack([b['depth_range'] for b in batch], dim=0)
    output['name'] = [b['name'] for b in batch]
    return output
