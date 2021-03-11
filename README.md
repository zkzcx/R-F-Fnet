## R FF network
paper

### VOC Dataset Test

|Method		|*pre-train*|*Training data*|	*Backbone network*|	*mAP*	|*fps*|
| :--------------------------------------- |:------: | :----------------------: | :-------------------------------: | :------: |:------: |
|Faster -RCNN	|√|	07 + 12	|VGGNet	|73.2	|7|
|R-FCN	|√|	07 + 12	|ResNet101|	80.5|	9|
|SSD300	|√|	07 + 12	|VGGNet|	77.5|	62|
|DSOD300	|×|	07 + 12| DS/64-192-48-1|	77.7|	17.4|
|GRP-DSOD320	|×|	07 + 12|	DS/64-192-48-1	|78.7|	16.7|
|FSSD300	|√|	07 + 12	|VGGNet|	78.8|	65.8|
|DSSD321	|√|	07 + 12	|ResNet101	|78.6|	9.5|
|RefineDet320	|√|	07 + 12|	VGGNet|	80.0|	40.3|
|ASIF-Det320	|√|	07 + 12|	VGGNet|	79.2|	33|
|RFB Net300	|√|	07 + 12	|VGGNet|	80.5|	83|
|ScratchDet300	|×|	07 + 12|	Root-ResNet-34|	80.4|	17.8|
|Our300	|√|	07 + 12	|VGGNet	|79.7|	83.3|
|SSD512	|√|	07 + 12	|VGGNet	|79.8	|26|
|FSSD512	|√|	07 + 12|	VGGNet|	80.9	|35.7|
|DSSD513	|√|	07 + 12|	ResNet101	|81.5|	5.5|
|RefineDet512	|√|	07 + 12	|VGGNet	|81.8|	24.1|
|RFB Net512	|√|	07 + 12	|VGGNet|	82.2	|38|
|Our512	|√|	07 + 12|	VGGNet|	81.8|	66.6|

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

###coco2017: images
|train|	118287|
| :----------------------------------- | :------: | 
|val	|5000|
|test|	40670|
###coco2014: images
|train|	82783|
|val	|40504|
|test|	40775|
##### train2017 == trainval35k == train2014 + val2014 - minival2014 == train2014 + val2014 - val2017
##### trainval == train2017 + val2017 == train2014 + val2014

### COCO Test 
### trainval vs trainval35k  (123; 287 V.S. 118; 287)
| Method	|*pre-train*|	*Training data*	|*Backbone network*	|*Time(ms)*	|*AP*	|*AP50*	|*AP75* |*APs*	|*APm*	|*APl*| 
| :--------------------------------------------------- | :------: |:-------------------------------: | :----------------------------------------------------------: | :------: |:------: |:------: |:------: |:------: |:------: |:------: |
|Faster-RCNN	    |√|	trainval35k	|VGG-16	|147	|24.2|	45.3	|23.5	|7.7	|26.4|	37.1|
|Faster-FPN	     |√|	trainval35k	|ResNet-101-FPN|240	|36.2	|59.1	|39.0	|18.2	|39.0	|48.2|
|R-FCN	     |√|	trainval|	ResNet-101	|110	|29.9|	51.9	|-|	10.8|	32.8|	45.0|
|RetinaNet-500	  |√|trainval35k|	ResNet-101	|90	|34.4|	53.1	|36.8	|14.7|	38.5	|49.1|
|SSD300		  |  √|trainval35k|	VGGNet|	12|	25.3	|42.0	|26.5|	6.2|	28.0	|43.3|
|DSOD300	     |×|	trainval	|DS/64-192-48-1	|-|	29.3|	47.3	|30.6|9.4	|31.5|	47.0|
|GRP-DSOD320    |×|trainval|	DS/64-192-48-1	|-	|30.0	|47.9	|31.8	|10.9	|33.6	|46.3|
|FSSD300	      |√|	trainval35k	|VGGNet|	16|	27.1|	47.7	|27.8|	8.7|	29.2|	42.2|
|DSSD321	      |√|	trainval35k|	ResNet-101|	-|	28.0|	46.1	|29.2|	7.4|	28.1|	47.6|
|RefineDet320      |√|		trainval35k	|VGGNet	|26|	29.4	|49.2|	31.3|	10.0|	32.0|	44.4|
|RFBNet300     |√|		trainval35k|	VGGNet|	15|	30.3|	49.3|	31.8|	11.8|	31.9|	45.9|
|ScratchDet300     |×| 	trainval35k|	Root-ResNet-34|	25|	32.7	|52.0|	34.9|	13.0|	35.6|	49.0|
|Our300        |√|	trainval35k|	VGGNet|	28|	28.0	|48.4	|28.3	|11.1|	31.4	|42.9|
|SSD512       |√|		trainval35k|	VGGNet|	28|	28.8|	48.5	|30.3|	10.9|	31.8	|43.5|
|FSSD512      |√|		trainval35k|	VGGNet|	36|	31.8	|52.8	|33.5	|14.2|	35.1	|45.0|
|DSSD513	    |√|	trainval35k|	ResNet-101|	156	|33.2|	53.3|	35.2|	13.0	|25.4|	51.1|
|RefineDet512      |√|		trainval35k|	VGGNet|	45|	33.0|	54.5	|35.5	|16.3	|36.3|	44.3|
|RFBNet512	     |√|	trainval35k|	VGGNet|	30|	33.8	|54.2|	35.9|	16.2|	37.1|	47.4|
|YOLO v3-608     |√|	trainval35k|	Darknet-53|	51|	33.0	|57.9	|34.4	|18.3	|35.4|	41.9|
|Our512 	     |√|	trainval35k|	VGGNet|	42	|33.1|	54.4|	35.1|	18.2|	36.5	|46.8|


## Installation
- Install [PyTorch 1.2.0](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository. This repository is mainly based on [lzx1413/PytorchSSD](https://github.com/lzx1413/PytorchSSD), and a huge thank to him.

- Compile the nms and coco tools:
```Shell
cd RFFnet
chmod 777 make.sh
bash make.sh
```
- Install pycocotool
```Shell
pip install pycocotool
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
python setup.py build_ext --inplace                # install pycocotools locally
python setup.py build_ext install                     # install pycocotools to the Python site-packages
```

## Datasets

### PASCAL VOC Dataset
##### Download PASCAL VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download PASCAL VOC2012 trainval

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```
##### Download MS COCO2017 Dataset
```Shell
http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip

http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

## http://images.cocodataset.org/zips/test2017.zip
## http://images.cocodataset.org/annotations/image_info_test2017.zip
```

## Training
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at: [BaiduYun Driver](https://pan.baidu.com/s/1F7sEPw1xtXJOCU7B6H8L-A 
), password is `hms4`.


```Shell
# Put vgg16_reducedfc.pth in a new folder 'weights' 
python train_test_vgg.py
```
### Personal advice

If you are interested in this paper or interested in lightweight detectors, please e-mail me
