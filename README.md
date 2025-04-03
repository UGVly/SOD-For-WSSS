# Salient Object Detection Enhanced Pseudo-Labels for Weakly Supervised Semantic Segmentation

This is the python implementation of our paper "Salient Object Detection Enhanced Pseudo-Labels for Weakly Supervised Semantic
 Segmentation", which is shows a method for Weakly Supervised Semantic Segmentation Method with Salient Object Detection method.

## Installations
```bash
git clone git@github.com:UGVly/SOD-For-WSSS.git
cd SOD-For-WSSS
```

## Requirements
* Platform preferences: Linux.
  Windows is also compatible, but the running speed will be slower due to the limit of the currently using parallel strategy.

The dependencies of this project are listed in `requirements.txt`. You can install them using the following command.
```bash
pip install -r requirements.txt
```

## Start
* SOD Data Prepare

For all datasets, they should be organized in below's fashion:
```
|__dataset_name
   |__Images: xxx.jpg ...
   |__Masks : xxx.png ...
```
For training, put your dataset folder under:
```
dataset/
```
For evaluation, download below datasets and place them under:
```
dataset/benchmark/
```
Suppose we use DUTS-TR for training, the overall folder structure should be:
```
|__dataset
   |__DUTS-TR
      |__Images: xxx.jpg ...
      |__Masks : xxx.png ...
   |__benchmark
      |__ECSSD
         |__Images: xxx.jpg ...
         |__Masks : xxx.png ...
      |__HKU-IS
         |__Images: xxx.jpg ...
         |__Masks : xxx.png ...
      ...
```
[**ECSSD**](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) || [**HKU-IS**](https://i.cs.hku.hk/~gbli/deep_saliency.html) || [**DUTS-TE**](http://saliencydetection.net/duts/) || [**DUT-OMRON**](http://saliencydetection.net/dut-omron/) || [**PASCAL-S**](http://cbi.gatech.edu/salobj/)


* WSSS Data Prepare
First download the Pascal VOC 2012 datasetsin the `data` dir.

```bash
cd data
```

Follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit


- Then download SBD annotations from [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip).
The folder structure is assumed to be:
```bash

|__ data
    |__ voc12
        |__ VOCdevkit
            |__ VOC2012
            |__ JPEGImages
            |__ SegmentationClass
            |__ SegmentationClassAug
|__voc12
    |__ cls_labels.npy
    |__ train_aug_id.txt
    |__ train_id.txt
    |__ val_id.txt
```

* Pretrain with SOD

```

python SOD_train.py --backbone swin-base-224 \
    --max_epoch 150 --lr 1e-4 --train_batch 16 --decay_epoch 15 \
    --gamma 0.5 --gpu_id 1 --train_root [sod data dir] \
    --log_path ./log/ --edge_loss 1 --cache-mode part --workers 4

```

* Train
  
To trigger the training process:
```

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_segmentation.py --backbone [resnest269 | resnest101] --mode fix --use_gn True --tag [task_name] --label_name [pslabel_ours] --data_dir [data dir]

```

For more adjustable configurations, please refer to `options.py`.

<!-- * Test

```
python run.py --test --load_path _path_to_checkpoint

``` -->

## Citing
```

```

## TODO
1. 
