# 3D Part Guided Image Editing for Fine-grained Object Understanding
<img src="https://github.com/zongdai/EditingForDNN/blob/master/image/Overview.jpg" width="860"/>

## Requirements
* python 3.6, cuda 9.2, pytorch 1.2.0, torchvision 0.4.0;
* python-opencv
## Usage
```
python tools/infer.py --pretrained_model ./pretrained_model/state_rcnn_double_backbone.pth --input_dir ./demo/imgs --output_dir ./demo/res
```
The pretrained model can be downloaded at [here](https://pan.baidu.com/s/1JzErnI4S0WV-ME4cNQd2xg) (code:owov)


<img src="https://github.com/zongdai/EditingForDNN/blob/master/image/infer_result.jpg" width="860"/>


## Editing Data
<img src="https://github.com/zongdai/EditingForDNN/blob/master/image/editing_images.jpg" width="860"/>

## CUS Dataset
<img src="https://github.com/zongdai/EditingForDNN/blob/master/image/CUS_images.jpg" width="860"/>
