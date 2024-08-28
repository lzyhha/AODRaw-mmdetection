# Towards RAW Object Detection in Diverse Conditions

[Paper link](https://arxiv.org/abs/2411.15678)

[Dataset link](https://github.com/lzyhha/AODRaw)

## Table of Contents
- [Introduction](#introduction)
- [Install](#install)
- [ModelZoo](#modelzoo)
   - [Models using down-sampled AODRaw](#models-using-down-sampled-aodraw)
   - [Models using sliced AODRaw](#models-using-sliced-aodraw)
- [Training and Evaluation](#training-and-evaluation)
   - [Configs and pre-trained weights](#configs-and-pre-trained-weights)
   - [Training](#training)
   - [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)
- [Acknowledgement](#acknowledgement)

## Introduction

Existing object detection methods often consider sRGB input, which was compressed from RAW data using ISP originally designed for visualization. However, such compression might lose crucial information for detection, especially under complex light and weather conditions. **We introduce the AODRaw dataset, which offers 7,785 high-resolution real RAW images with 135,601 annotated instances spanning 62 categories, capturing a broad range of indoor and outdoor scenes under 9 distinct light and weather conditions.** Based on AODRaw that supports RAW and sRGB object detection, we provide a comprehensive benchmark for evaluating current detection methods. We find that sRGB pre-training constrains the potential of RAW object detection due to the domain gap between sRGB and RAW, prompting us to directly pre-train on the RAW domain. However, it is harder for RAW pre-training to learn rich representations than sRGB pre-training due to the camera noise. To assist RAW pre-training, we distill the knowledge from an off-the-shelf model pre-trained on the sRGB domain. As a result, we achieve substantial improvements under diverse and adverse conditions without relying on extra pre-processing modules. 

## Dataset

Please refer to [AODRaw]() to download and preprocess our AODRaw dataset.

## Install

Please refer to the [README of mmdetection](README_MMDET.md).

## ModelZoo

#### Models using down-sampled AODRaw

Please follow [downsampling](https://github.com/lzyhha/AODRaw/tree/main?tab=readme-ov-file#down-sampling-precrossing) to preprocess the images or download preprocessed images in [download](https://github.com/lzyhha/AODRaw/tree/main?tab=readme-ov-file#dataset-and-downloading).

|  Detector |Backbone | Pre-training domain | Fine-tuning domain | AP | Config | Model | Pre-trained weights |
|  ---------------------  | -------------------- | :--------------------: | :--------------------: | :--------------------: | :--------------------: |  :--------------------: |:--------------------:|
|Faster RCNN | ResNet-50 | sRGB | sRGB | 23.3 | [Config](configs/aodraw/faster-rcnn_r50_fpn_1x_aodraw_srgb.py) | [Google](https://drive.google.com/file/d/1dUHyWXNLdg4WW165tvbsDp5D7G0OjD0Q/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1XPipT0uMl8mqxUjjG5EoWA?pwd=w251) | - |
|Retinanet | ResNet-50 | sRGB | sRGB | 19.1 | [Config](configs/aodraw/retinanet_r50_fpn_1x_aodraw_srgb.py) | [Google](https://drive.google.com/file/d/1CFTCP1VVgx3pfA-ITE2S2kJ6C3H6pqdW/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/19JoTMeaTKOcJOcKzVnRQWg?pwd=qvxf) | - |
|GFocal | ResNet-50 | sRGB | sRGB | 24.2 | [Config](configs/aodraw/gfl_r50_fpn_1x_aodraw_srgb.py) | [Google](https://drive.google.com/file/d/1HRCvJJyQUpWo9-XQaG93BQ4HReQUaPYI/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1iMHk3eM5gSJqUSG7zDNm3g?pwd=se68) | - |
|Sparse RCNN | ResNet-50 | sRGB | sRGB | 15.6 | [Config](configs/aodraw/sparse-rcnn_r50_fpn_1x_aodraw_srgb.py) | [Google](https://drive.google.com/file/d/11TKxMNblfzHQKWvQ_e2UDxzX1ZSkhEbw/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1HBAbtZghftRDytq_VuNqvQ?pwd=wy8j) | - |
|Deformable DETR | ResNet-50 | sRGB | sRGB | 16.6 | [confog](configs/aodraw/deformable-detr_r50_16xb2-100e_aodraw_srgb.py) | [Google](https://drive.google.com/file/d/1ToaCFi0rXe5RdwTkh9OIh9RxlIxtyydC/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/154F3nnSGFEujZBOHGGPsMQ?pwd=vbct) | - |
|Cascade RCNN| ResNet-50 | sRGB | sRGB | 25.6 | [Config](configs/aodraw/cascade-rcnn_r50_fpn_1x_aodraw_srgb.py) | [Google](https://drive.google.com/file/d/1x9ggEXeeTMcHrt42ycWzn7oJvgO4CHj3/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/166HVewRzxweHidfilEqSwA?pwd=7hgm) |  - |
|Faster RCNN | Swin-T | sRGB |sRGB | 28.4 | [Config](configs/aodraw/faster-rcnn_swin-t-p4-w7_fpn_1x_aodraw_srgb.py) | [Google](https://drive.google.com/file/d/14uryTqn4PfSMHll-3jVoY3dfv6BMeBpV/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1OIMEOzdEoO4zQWKfnEIWeg?pwd=dzah) | - |
|Faster RCNN | ConvNeXt-T | sRGB | sRGB | 29.7 | [Config](configs/aodraw/faster-rcnn_convnext-t-p4-w7_fpn_amp-1x_aodraw_srgb.py) | [Google](https://drive.google.com/file/d/1OjNIZ42mlPaMwy10BA4NVnCs9jqPWYTc/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1EKc9gdKqrrWHhCI2qgj9bw?pwd=ibvy) | [Google](https://drive.google.com/file/d/12R1-QcqMyjVo66nOp5NtK9-tSzj11SV3/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1js44KwmeD4dQGY29zreQaQ?pwd=vecd)   |
|GFocal | Swin-T | sRGB | sRGB | 30.1 | [Config](configs/aodraw/gfl_swin-t-p4-w7_fpn_1x_aodraw_srgb.py) | [Google](https://drive.google.com/file/d/1jPVq-gkNRkFAqM1HKLWmHBxi5i7hq8l0/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/16rdWnleYcxR6pnm2k3f7Yw?pwd=vhe7) | - |
|GFocal | ConvNeXt-T | sRGB | sRGB | 32.1 | [Config](configs/aodraw/gfl_convnext-t-p4-w7_fpn_1x_aodraw_srgb.py) | [Google](https://drive.google.com/file/d/1YMasTjI53OSWToD1btC2o25XYt36uiur/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1tatP2KyudFerkB3_WHyqDQ?pwd=zpws) | [Google](https://drive.google.com/file/d/12R1-QcqMyjVo66nOp5NtK9-tSzj11SV3/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1js44KwmeD4dQGY29zreQaQ?pwd=vecd)   |
|Cascade RCNN | Swin-T | sRGB | sRGB | 32.0 | [Config](configs/aodraw/cascade-rcnn_swin-t-p4-w7_fpn_4conv1fc-giou_amp-1x_aodraw_srgb.py) | [Google](https://drive.google.com/file/d/1yZ93gIuogxUU8eRenJ5eEFykrdYyPAW7/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1JeoZfTMLWh1JRX4CcUlWmg?pwd=sbpk) |  - |
|Cascade RCNN | ConvNeXt-T | sRGB | sRGB | 34.0 | [Config](configs/aodraw/cascade-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-1x_aodraw_srgb.py) | [Google](https://drive.google.com/file/d/1hf9G3LYrGWd_37CIJzZZr_TBjbuFfGyH/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1OqJ9h4Fot8QJbQiD2gc4Ng?pwd=2kud) | [Google](https://drive.google.com/file/d/12R1-QcqMyjVo66nOp5NtK9-tSzj11SV3/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1js44KwmeD4dQGY29zreQaQ?pwd=vecd)   |

The directory [images_downsampled_srgb](https://github.com/lzyhha/AODRaw/tree/main?tab=readme-ov-file#dataset-and-downloading) is required for the above experiments.

|  Detector |Backbone | Pre-training domain | Fine-tuning domain | AP | Config | Model | Pre-trained weights |
|  ---------------------  | -------------------- | :--------------------: | :--------------------: | :--------------------: | :--------------------: |  :--------------------: |:--------------------:|
|GFocal | Swin-T | sRGB | RAW | 29.9 |[Config](configs/aodraw/gfl_swin-t-p4-w7_fpn_1x_aodraw_raw.py) | [Google](https://drive.google.com/file/d/1x_uX3wfI1s2qU7ILe9TOy1ikcCyfI8lf/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1Rw7ebbJm93WdQTuz5gOeCQ?pwd=3xcd) | - |
|GFocal | ConvNeXt-T | sRGB | RAW | 31.5 | [Config](configs/aodraw/gfl_convnext-t-p4-w7_fpn_1x_aodraw_raw.py) | [Google](https://drive.google.com/file/d/159ENxlvyP-3mJ8sDDQ5l-b3vOSsvG-XR/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1iqHJe4RPKmjwDM2rzc_CVg?pwd=xgv9) |[Google](https://drive.google.com/file/d/12R1-QcqMyjVo66nOp5NtK9-tSzj11SV3/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1js44KwmeD4dQGY29zreQaQ?pwd=vecd)   |
|Cascade RCNN | Swin-T | sRGB | RAW | 31.7 | [Config](configs/aodraw/cascade-rcnn_swin-t-p4-w7_fpn_4conv1fc-giou_amp-1x_aodraw_raw.py) | [Google](https://drive.google.com/file/d/1C8XEdLOw8b-K7cscTj9964DKplNK1udQ/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1_UGoMxofGI-cIMSS4JnzqQ?pwd=yf1u) | - |
|Cascade RCNN | ConvNeXt-T | sRGB | RAW | 33.7 | [Config](configs/aodraw/cascade-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-1x_aodraw_raw.py) | [Google](https://drive.google.com/file/d/15K0wNjlMK1QkQPXblXp-Pced4dLIynYV/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1lIGgUz_AWwPBu3b-luHjdg?pwd=er6c)|[Google](https://drive.google.com/file/d/12R1-QcqMyjVo66nOp5NtK9-tSzj11SV3/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1js44KwmeD4dQGY29zreQaQ?pwd=vecd)   |

The directory [images_downsampled_raw](https://github.com/lzyhha/AODRaw/tree/main?tab=readme-ov-file#dataset-and-downloading) is required for the above experiments.

|  Detector |Backbone | Pre-training domain | Fine-tuning domain | AP | Config | Model | Pre-trained weights |
|  ---------------------  | -------------------- | :--------------------: | :--------------------: | :--------------------: | :--------------------: |  :--------------------: |:--------------------:|
|GFocal | Swin-T | RAW | RAW | 30.7 | [Config](configs/aodraw/gfl_swin-t-p4-w7_fpn_1x_aodraw_raw_raw-pretraining.py) | [Google](https://drive.google.com/file/d/18e8cnEsQOEjdp1N99Sqnn_ZHofs0fjcY/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1AolwF5_1uZ1Xhw8E5ibKSA?pwd=s3vi) | [Google](https://drive.google.com/file/d/12hdeZMp6cn4dKIidL07ndY1xw59qAnbO/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1mCrunp0rrFUAlMrxiui9mQ?pwd=nm1r)  |
|GFocal | ConvNeXt-T | RAW | RAW | 32.1 | [Config](configs/aodraw/gfl_convnext-t-p4-w7_fpn_1x_aodraw_raw_raw-pretraining.py) | [Google](https://drive.google.com/file/d/1cRtbbsSpokYcp_dte-YQDeeIB_n2A40F/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1eaqVwbTLnsIMFYMCXUHRdg?pwd=r7xu) | [Google](https://drive.google.com/file/d/1U9KK7-PcWIxbDPSUs6bx2ig7q_9NX_KZ/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1van4UUYqL90w9VHk68fC3A?pwd=9262) | 
|Cascade RCNN | Swin-T | RAW | RAW | 32.2 | [Config](configs/aodraw/cascade-rcnn_swin-t-p4-w7_fpn_4conv1fc-giou_amp-1x_aodraw_raw_raw-pretraining.py) | [Google](https://drive.google.com/file/d/1BxuYMtKhWphoaGcH-UC7BzQYc45ZiW73/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1EpTn7eQQB-6v4be-B17vHg?pwd=7kfs) | [Google](https://drive.google.com/file/d/12hdeZMp6cn4dKIidL07ndY1xw59qAnbO/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1mCrunp0rrFUAlMrxiui9mQ?pwd=nm1r)  | 
|Cascade RCNN | ConvNeXt-T | RAW | RAW | 34.8 | [Config](configs/aodraw/cascade-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-1x_aodraw_raw_raw-pretraining.py) | [Google](https://drive.google.com/file/d/1nfwyfLK3nQ6cjGXkuHdFKRlEx1WJUgN_/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1FOda_Ipw17kHxWTLoqMXHA?pwd=kh1b) |  [Google](https://drive.google.com/file/d/1U9KK7-PcWIxbDPSUs6bx2ig7q_9NX_KZ/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1van4UUYqL90w9VHk68fC3A?pwd=9262) |

The directory [images_downsampled_raw](https://github.com/lzyhha/AODRaw/tree/main?tab=readme-ov-file#dataset-and-downloading) is required for the above experiments.

#### Models using sliced AODRaw
Please follow [slicing](https://github.com/lzyhha/AODRaw/tree/main?tab=readme-ov-file#slicing-precrossing) to preprocess the images or download preprocessed images in [download](https://github.com/lzyhha/AODRaw/tree/main?tab=readme-ov-file#dataset-and-downloading).


|  Detector |Backbone | Pre-training domain | Fine-tuning domain | AP | Config | Model | Pre-trained weights |
|  ---------------------  | -------------------- | :--------------------: | :--------------------: | :--------------------: | :--------------------: |  :--------------------: |:--------------------:|
|Cascade RCNN | Swin-T | sRGB | RAW | 29.2 | [Config](configs/aodraw_slice/cascade-rcnn_swin-t-p4-w7_fpn_4conv1fc-giou_amp-1x_aodraw_raw_slice.py) | [Google](https://drive.google.com/file/d/1jNbChm9eJ4NJaMZioFfFqNGBx2KFx2Dh/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1t1EeOIblW9d4nhRc7hrPQw?pwd=qis7) | - |
|Cascade RCNN | ConvNeXt-T | sRGB | RAW | 29.7 | [Config](configs/aodraw_slice/cascade-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-1x_aodraw_raw_slice.py) | [Google](https://drive.google.com/file/d/1TF08ZVywXN5nM8jGvfespbxfb9qTJI86/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1UY1YtbyXqK5zFcbsLxYNGA?pwd=q7px) |[Google](https://drive.google.com/file/d/12R1-QcqMyjVo66nOp5NtK9-tSzj11SV3/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1js44KwmeD4dQGY29zreQaQ?pwd=vecd)   |

The directory [images_slice_raw](https://github.com/lzyhha/AODRaw/tree/main?tab=readme-ov-file#dataset-and-downloading) is required for the above experiments.

|  Detector |Backbone | Pre-training domain | Fine-tuning domain | AP | Config | Model | Pre-trained weights |
|  ---------------------  | -------------------- | :--------------------: | :--------------------: | :--------------------: | :--------------------: |  :--------------------: |:--------------------:|
|Cascade RCNN | Swin-T | RAW | RAW| 29.8 | [Config](configs/aodraw_slice/cascade-rcnn_swin-t-p4-w7_fpn_4conv1fc-giou_amp-1x_aodraw_raw_slice_raw-pretraining.py) | [Google](https://drive.google.com/file/d/1h2ahPnBftYITcrPtsKlT7ma_ST5E3P0C/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1SfpWp4EXThhJO-zQqoAJog?pwd=5k49) | - |
|Cascade RCNN | ConvNeXt-T | RAW | RAW | 30.7 | [Config](configs/aodraw_slice/cascade-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-1x_aodraw_raw_slice_raw-pretraining.py) | [Google](https://drive.google.com/file/d/1w1zlbPoCWeG3iYm34sJbRpUpCs8bdpvr/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1T7lY-JKb2B9javuQRROMgA?pwd=73mp) | [Google](https://drive.google.com/file/d/1U9KK7-PcWIxbDPSUs6bx2ig7q_9NX_KZ/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1van4UUYqL90w9VHk68fC3A?pwd=9262) |

The directory [images_slice_raw](https://github.com/lzyhha/AODRaw/tree/main?tab=readme-ov-file#dataset-and-downloading) is required for the above experiments.

## Training and Evaluation

#### Configs and pre-trained weights

- We provide training and evaluation for RAW and sRGB object detection. 
- The images in the AODRaw are recorded at a resolution of $6000\times 4000$. It is unrealistic to feed such huge images into the detectors. Thus, we adopt two experiment settings: 1) down-sampling the images into a lower resolution of $2000\times1333$, corresponding to [configs](#models-using-down-sampled-aodraw), and 2) slicing the images into a collection of $1280\times 1280$ patches, corresponding to [configs](#models-using-sliced-aodraw). **Please preprocess the AODRaw dataset or directly download the processed files in [datasets and downloading](https://github.com/lzyhha/AODRaw/tree/main?tab=readme-ov-file#dataset-and-downloading).**

Training and evaluation using down-sampled AODRaw:
|  Task | Pre-training domain | Config path |
|  ---------------------  | -------------------- |-------------------- |
| sRGB object detection | sRGB | configs/aodraw/..._aodraw_srgb.py |
| RAW object detection | sRGB | configs/aodraw/..._aodraw_raw.py |
| RAW object detection | RAW | configs/aodraw/..._aodraw_raw_raw-pretraining.py |

Training and evaluation using sliced AODRaw:
|  Task | Pre-training | Config path |
|  ---------------------  | -------------------- |-------------------- |
| sRGB object detection | sRGB | configs/aodraw_slice/..._aodraw_srgb_slice.py |
| RAW object detection | sRGB | configs/aodraw_slice/..._aodraw_raw_slice.py |
| RAW object detection | RAW | configs/aodraw_slice/..._aodraw_raw_slice_raw-pretraining.py |

Pre-trained weights for ConvNeXt-T and Swin-T:
|  Architecture | Pre-training domain | Downloading link |
|  ---------------------  | -------------------- |-------------------- |
| ConvNeXt-T | sRGB | [Google](https://drive.google.com/file/d/12R1-QcqMyjVo66nOp5NtK9-tSzj11SV3/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1js44KwmeD4dQGY29zreQaQ?pwd=vecd)   |
| ConvNeXt-T | RAW |  [Google](https://drive.google.com/file/d/1U9KK7-PcWIxbDPSUs6bx2ig7q_9NX_KZ/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1van4UUYqL90w9VHk68fC3A?pwd=9262) |
| Swin-T | RAW |  [Google](https://drive.google.com/file/d/12hdeZMp6cn4dKIidL07ndY1xw59qAnbO/view?usp=sharing) and [Baidu](https://pan.baidu.com/s/1mCrunp0rrFUAlMrxiui9mQ?pwd=nm1r)  |

#### Training

##### Single GPU

   ```shell
   python tools/train.py ${CONFIG_FILE} [optional arguments]
   ```

##### Multi GPU

   ```shell
   bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
   ```

For more training and evaluation command details, please refer to [mmdetection](https://github.com/open-mmlab/mmdetection?tab=readme-ov-file#getting-started).

#### Evaluation

##### Single GPU

   ```shell
   python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
   ```

##### Multi GPU

   ```shell
   bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
   ```

For more training and evaluation command details, please refer to [mmdetection](https://github.com/open-mmlab/mmdetection?tab=readme-ov-file#getting-started).

## Citation
```
@article{li2024aodraw,
  title={Towards RAW Object Detection in Diverse Conditions}, 
  author={Zhong-Yu Li and Xin Jin and Boyuan Sun and Chun-Le Guo and Ming-Ming Cheng},
  journal={arXiv preprint arXiv:2411.15678},
  year={2024},
}
```

## License

The code is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for NonCommercial use only.

## Acknowledgement

This repo is modified from [mmdetection](https://github.com/open-mmlab/mmdetection).

