# Unsupervised-Optical-Flow

<p align="center">
  <img src="teaser/fc0768f4d5a34d12.gif" alt="example input-output gif" width="512" />
</p>
 
 ## Create dataset
 ```shell
 mkdir raw_dataset
 ```
 Copy the video recodings in the "raw_dataset" directory. A sample of the UPB dataset is available <a href="https://drive.google.com/drive/folders/1p_2-_Xo-Wd9MCnkYqPfGyKs2BnbeApqn?usp=sharing">here</a>.
 
 Split the videos in train and test/validation:
 
 ```shell
 python3 scripts/split_dataset.py \
  --src_dir raw_dataset
  --dst_dir split_scenes
 ```
 
 Transform videos into frames for training:
 
 ```shell
 python3 scripts/create_dataset.py \
  --src_dir raw_dataset\
  --dst_dir dataset \
  --split_dir split_scenes \
  --dataset upb
```

## Train model

``` shell
python3 train.py \
 --dataset upb \
 --batch_size 12 \
 --num_epochs 20 \
 --scheduler_step_size 15 \
 --height 256 \
 --width 512 \
 --model default
```

## Pre-trained model
```shell
mkdir -p snapshots_simple/checkpoints/
```
Download pretrained model from <a href="https://drive.google.com/file/d/1CIwSjlzHW3IlctN_1dfNSIBBLxoUGDix/view?usp=sharing">here</a> into the "snapshots/checkpoints" directory


## Test model

The checkpoint should be present in "snaphsots_simple/checkpoints" directory

```shell
python3 test.py \
 --video_path raw_dataset/fc0768f4d5a34d12.mov \
 --save_gif teaser/fc0768f4d5a34d12.gif \
 --model_name default.pth
```
