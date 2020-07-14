## RFFNet
A Rich Feature Fusion Single-Stage Object Detector

### COCO Test
|Method	|*Time*	 |*AP*	 |*APs* |
| :--------------------------------------- | :------: | :-----------------------: | :------: |
|SSD512	|28	 |28.8	 |10.9 |
|FSSD512	|35	 |31.8	 |14.2 |
|DSOD300	|57.5	 |29.3	 |9.4 |
|YOLO v3-608	|51	 |33.0	 |18.3 |
|DSSD513	|156	 |33.2	 |13.0 |
|RefineDet512	|42	 |33.0	 |16.3 |
|RetinaNet-500	|73	 |32.5	 |13.9 |
|ScratchDet300	|25	 |32.7	 |13.0 |
|RFBNet512	|30	 |33.8	 |16.2 |
|Our300	 |28	|28.0	 |11.1 |
|Our512	 |42	|33.1	 |18.2 |

## Installation
- Install [PyTorch 1.2.0](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository. This repository is mainly based on[lzx1413/PytorchSSD](https://github.com/lzx1413/PytorchSSD), and a huge thank to him.

- Compile the nms and coco tools:
```Shell
./make.sh
```

## Datasets

### VOC Dataset
##### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## Training
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at: [BaiduYun Driver](https://pan.baidu.com/s/1nzOgaL8mAPex8_HLU4mb8Q), password is `mu59`.


```Shell
# Put vgg16_reducedfc.pth in a new folder weights and 
python train_test_mob.py or python train_test_vgg.py
```
### Personal advice: when use Mobilenet v1 to train voc datasets, use a higher learning rate at the beginning, the convergence performance may be better.

If you are interested in this paper or interested in lightweight detectors, please QQ me (1006626961)
