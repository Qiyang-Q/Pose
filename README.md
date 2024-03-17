# 6D Pose Estimation

## Environment
- Pytorch Lightning: 2.1.2
- Pytorch: 2.0.1
(see more from requirements.txt)

## Training and Evaluation on OnePose dataset
### Dataset setup 
1. Download OnePose dataset from [onedrive storage](https://zjueducn-my.sharepoint.com/:f:/g/personal/zihaowang_zju_edu_cn/ElfzHE0sTXxNndx6uDLWlbYB-2zWuLfjNr56WxF11_DwSg?e=GKI0Df) and extract them into `$/your/path/to/onepose_datasets`. 
The directory should be organized in the following structure:
    ```
    |--- /your/path/to/onepose_datasets
    |       |--- train_data
    |       |--- val_data
    |       |--- test_data
    |       |--- sample_data
    ```

2. Build the dataset symlinks
    ```shell
    REPO_ROOT=/path/to/OnePose
    ln -s /your/path/to/onepose_datasets $REPO_ROOT/data/onepose_datasets
    ```

3. Run Structure-from-Motion for the data sequences (for train and inference on OnePose)

    Reconstructed the object point cloud and 2D-3D correspondences are needed for both training and test objects: (In our part we only use these steps to filter out images with wrong poses, okay to skip)
    ```python
    python run.py +preprocess=sfm_spp_spg_train.yaml # for training data
    python run.py +preprocess=sfm_spp_spg_test.yaml # for testing data
    python run.py +preprocess=sfm_spp_spg_val.yaml # for val data
    python run.py +preprocess=sfm_spp_spg_sample.yaml # an example, if you don't want to test the full dataset
    ```

### Begin training
1. Prepare ground-truth annotations. Merge annotations of training/val data: 
    ```python
    python run.py +preprocess=merge_anno task_name=onepose split=train
    python run.py +preprocess=merge_anno task_name=onepose split=val
    ```
   
2. Train NeRF
    ```python
    python train.py +experiment=NeRF.yaml exp_name=training_onepose
    ```
    At the end of training stage, model will build voxel and image to voxel's correspondence and save to dataset folder

3. Train Pose
    ```python
    python train_pose.py +experiment=pose.yaml exp_name=training_onepose
    ```

### Test on Onepose Dataset
    ```python
    python test_pose.py +experiment=pose_test_seen_obj.yaml exp_name=training_onepose
    ```
   
All model weights will be saved under `${REPO_ROOT}/data/models/checkpoints/${exp_name}` and logs will be saved under `${REPO_ROOT}/data/logs/${exp_name}`.
<!-- You can visualize the training process by tensorboard:
```shell
tensorboard xx
``` -->
