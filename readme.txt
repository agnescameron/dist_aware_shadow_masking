
## Requirement
 Python 3.5
 PyTorch 0.4.1
 torchvision
 numpy
 Cython
 pydensecrf (https://github.com/Andrew-Qibin/dss_crf)

## Path Setting
 You can modify the path in config.py

## Training
 1. First downloading the pretrained ReseNext model. (https://drive.google.com/file/d/1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ/view) and put it into ./resnext/
 2. Download the Dataset. Here we only provide our distraction datasets, since you can download the other datasets from the corresponding official website.
 2. python train.py

## Testing
 1. Download the pretrained model (see our project page) and put them in ckpt/models
 2. run corresponding testing code. (i.e., test_sbu.py, test_istd.py, test_ucf.py)


# Download 
 Please download the mentioned models or dataset from the project page. (https://quanlzheng.github.io/projects/Distraction-aware-Shadow-Detection.html)


#### Note that the provided model is a bit differnet with the one in the paper, since I mis-deleted the original models by accidencent and re-retrained the model. 
 
 I list the quantitative resutls of the provided model for reference.
 SBU: 3.27 | 2.49 | 4.04 
 UCF: 6.96 | 7.68 | 6.23
 ISTD: 1.74| 0.68 | 2.79


## Acknowlegement
 Part of the code is based upon BDRAR (https://github.com/zijundeng/BDRAR).

## Citation
 If you find this work useful for your research, please cite:

@InProceedings{Zheng_2019_CVPR,
author = {Zheng, Quanlong and Qiao, Xiaotian and Cao, Ying and Lau, Rynson W.H.},
title = {Distraction-Aware Shadow Detection},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}


## Contact
 This code is implemented by Quanlong Zheng. If you have any problem, please contact me via the email: xiaolong921001@gmail.com



