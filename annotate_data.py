import json
import os
import glob
import hydra

import os.path as osp
from loguru import logger
from pathlib import Path
from omegaconf import DictConfig

def get_filter_dict():
    filter_dict = {'0443': {'2': ['210', '470', '-1']},
                   '0448': {'3': ['605', '700']},
                   '0460': {'2': ['540', '585', '615', '620']},
                   '0463': {'2': ['355', '360']},
                   '0464': {'3': ['520']},
                   '0477': {'3': ['400']},
                   '0499': {'2': ['635']},
                   '0529': {'3': ['415', '830', '-1']},
                   '0536': {'1': ['435', '705', '-1'], '2': ['520', '960', '-1'], '3': ['420', '650', '-1']},
                   '0545': {'1': ['805']},
                   '0561': {'1': ['495'], '2': ['745', '750', '965']},
                   }
    for catagory_id in filter_dict.keys():
        for traj_id in filter_dict[catagory_id].keys():
            if filter_dict[catagory_id][traj_id][-1] == '-1':
                filter_dict[catagory_id][traj_id] = [str(i) for i in range(int(filter_dict[catagory_id][traj_id][0]),
                                                                             int(filter_dict[catagory_id][traj_id][1])+1, 5)]
    return filter_dict
def filter_anno(anno):
    filter_dict = get_filter_dict()
    catagory_idx = anno['pose_file'].split('/')[-4].split('-')[0]
    sequence_idx = anno['pose_file'].split('/')[-3].split('-')[-1]
    file_idx = anno['pose_file'].split('/')[-1].split('.')[0]
    if sequence_idx == '1' and file_idx == '410':
        x=1
    if catagory_idx not in filter_dict.keys():
        return False
    if sequence_idx not in filter_dict[catagory_idx].keys():
        return False
    if file_idx in filter_dict[catagory_idx][sequence_idx]:
        return True
    return False

def merge_(anno_2d_file, avg_anno_3d_file, collect_anno_3d_file,
           idxs_file, img_id, ann_id, images, annotations):
    """ To prepare training and test objects, we merge annotations about difference objs"""
    with open(anno_2d_file, 'r') as f:
        annos_2d = json.load(f)
    catagory_range = [ann_id + 1, ann_id + len(annos_2d) + 1]
    for anno_2d in annos_2d:
        if filter_anno(anno_2d):
            continue
        img_id += 1
        info = {
            'id': img_id,
            'img_file': anno_2d['img_file'],
        }
        images.append(info)


        ann_id += 1
        anno = {
            'image_id': img_id,
            'id': ann_id,
            'pose_file': anno_2d['pose_file'],
            'intrinsic_file': anno_2d['pose_file'].replace('poses_ba', 'intrin_ba'),
            'mask_file': anno_2d['pose_file'].replace('poses_ba', 'mask').replace('.txt', '.json'),
            'anno2d_file': anno_2d['anno_file'],
            'avg_anno3d_file': avg_anno_3d_file,
            'collect_anno3d_file': collect_anno_3d_file,
            'idxs_file': idxs_file,
            'catagory_range': catagory_range
        }
        annotations.append(anno)
    return img_id, ann_id


def merge_anno(cfg):
    """ Merge different objects' anno file into one anno file """
    anno_dirs = []

    if cfg.split == 'train':
        names = cfg.train.names
    elif cfg.split == 'val':
        names = cfg.val.names
    elif cfg.split == 'test':
        names = cfg.test.names

    for name in names:
        anno_dir = osp.join(cfg.datamodule.data_dir, name, f'outputs_{cfg.network.detection}_{cfg.network.matching}',
                            'anno')
        anno_dirs.append(anno_dir)

    img_id = 0
    ann_id = 0
    images = []
    annotations = []
    for anno_dir in anno_dirs:
        logger.info(f'Merging anno dir: {anno_dir}')
        anno_2d_file = osp.join(anno_dir, 'anno_2d.json')
        avg_anno_3d_file = osp.join(anno_dir, 'anno_3d_average.npz')
        collect_anno_3d_file = osp.join(anno_dir, 'anno_3d_collect.npz')
        idxs_file = osp.join(anno_dir, 'idxs.npy')

        if not osp.isfile(anno_2d_file) or not osp.isfile(avg_anno_3d_file) or not osp.isfile(collect_anno_3d_file):
            logger.info(f'No annotation in: {anno_dir}')
            continue

        img_id, ann_id = merge_(anno_2d_file, avg_anno_3d_file, collect_anno_3d_file,
                                idxs_file, img_id, ann_id, images, annotations)

    logger.info(f'Total num: {len(images)}')
    instance = {'images': images, 'annotations': annotations}

    out_dir = osp.dirname(cfg.datamodule.out_path)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    with open(cfg.datamodule.out_path, 'w') as f:
        json.dump(instance, f)


@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()