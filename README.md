# 6D Pose Estimation

## Environment
-Pytorch Lightning: 2.1.2
-Pytorch: 2.0.1
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

    Reconstructed the object point cloud and 2D-3D correspondences are needed for both training and test objects:
    ```python
    python run.py +preprocess=sfm_spp_spg_train.yaml # for training data
    python run.py +preprocess=sfm_spp_spg_test.yaml # for testing data
    python run.py +preprocess=sfm_spp_spg_val.yaml # for val data
    python run.py +preprocess=sfm_spp_spg_sample.yaml # an example, if you don't want to test the full dataset
    ```

### Inference on OnePose dataset
1. Download the pretrain weights [pretrained model](https://drive.google.com/drive/folders/1VjLLjJ9oxjKV5Xy3Aty0uQUVwyEhgtIE?usp=sharing) and move it to `${REPO_ROOT}/data/model/checkpoints/onepose/GATsSPG.ckpt`.

2. Inference with category-agnostic 2D object detection.

    When deploying OnePose to a real world system, 
    an off-the-shelf category-level 2D object detector like [YOLOv5](https://github.com/ultralytics/yolov5) can be used.
    However, this could defeat the category-agnostic nature of OnePose.
    We can instead use a feature-matching-based pipeline for 2D object detection, which locates the scanned object on the query image through 2D feature matching.
    Note that the 2D object detection is only necessary during the initialization.
    After the initialization, the 2D bounding box can be obtained from projecting the previously detected 3D bounding box to the current camera frame.
    Please refer to the [supplementary material](https://zju3dv.github.io/onepose/files/onepose_supp.pdf) for more details. 

    ```python
    # Obtaining category-agnostic 2D object detection results first.
    # Increasing the `n_ref_view` will improve the detection robustness but with the cost of slowing down the initialization speed.
    python feature_matching_object_detector.py +experiment=object_detector.yaml n_ref_view=15

    # Running pose estimation with `object_detect_mode` set to `feature_matching`.
    # Note that enabling visualization will slow down the inference.
    python inference.py +experiment=test_GATsSPG.yaml object_detect_mode=feature_matching save_wis3d=False
    ```

3. Running inference with ground-truth 2D bounding boxes

    The following command should reproduce results in the paper, which use 2D boxes projected from 3D boxes as object detection results.

    ```python
    # Note that enabling visualization will slow down the inference.
    python inference.py +experiment=test_GATsSPG.yaml object_detect_mode=GT_box save_wis3d=False # for testing data
    ```


### Training the GATs Network
1. Prepare ground-truth annotations. Merge annotations of training/val data:
    ```python
    python run.py +preprocess=merge_anno task_name=onepose split=train
    python run.py +preprocess=merge_anno task_name=onepose split=val
    ```
   
2. Begin training
    ```python
    python train.py +experiment=train_GATsSPG task_name=onepose exp_name=training_onepose
    ```
   
All model weights will be saved under `${REPO_ROOT}/data/models/checkpoints/${exp_name}` and logs will be saved under `${REPO_ROOT}/data/logs/${exp_name}`.
<!-- You can visualize the training process by tensorboard:
```shell
tensorboard xx
``` -->
