# 3D Part Guided Image Editing for Fine-grained Object Understanding
Zongdai Liu, Feixiang Lu, Peng Wang, Hui Miao, Liangjun Zhang, Ruigang Yang, Bin Zhou
## Abstrct
Holistically understanding an object with its 3D movable parts is essential for visual models of a robot to interact with the world. For example, only by understanding many possible part dynamics of other vehicles (e.g., door or trunk opening, taillight blinking for changing lane), a self-driving vehicle can be success in dealing with emergency cases. However, existing visual models tackle rarely on these situations, but focus on bounding box detection. In this paper, we fill this important missing piece in autonomous driving by solving two critical issues. First, for dealing with data scarcity, we propose an effective training data generation process by fitting a 3D car model with dynamic parts to cars in real images. This allows us to directly edit the real images using the aligned 3D parts, yielding effective training data for learning robust deep neural networks (DNNs). Secondly, to benchmark the quality of 3D part understanding, we collected a large dataset in real driving scenario with cars in uncommon states (CUS), i.e. with door or trunk opened etc., which demonstrates that our trained network with edited images largely outperforms other baselines in terms of 2D detection and instance segmentation accuracy.


<img src="https://github.com/zongdai/EditingForDNN/blob/master/image/Overview.jpg" width="860"/>

## Requirements
* python 3.6, cuda 9.2, pytorch 1.2.0, torchvision 0.4.0;
* python-opencv, pycocotools
## Inferring
```
python tool/infer.py --pretrained_model ./pretrained_model/state_rcnn_double_backbone.pth --input_dir ./demo/imgs --output_dir ./demo/res
```
The pretrained model can be downloaded at [here](https://pan.baidu.com/s/1JzErnI4S0WV-ME4cNQd2xg) (password:owov)


<img src="https://github.com/zongdai/EditingForDNN/blob/master/image/infer_result.jpg" width="860"/>


## Training
The editing data totally 27k could be downloaded at [here](https://pan.baidu.com/s/1Z5rBC9Jr-Fa22bTiJ7PiEQ)(password:6smu)
<img src="https://github.com/zongdai/EditingForDNN/blob/master/image/editing_images.jpg" width="860"/>
Download the editing data and place a softlink (or the actual data) in EditingForDNN/editing_data/.
```
cd EditingForDNN
mkdir editing_data
ln -s /path/images ./editing_data/
ln -s /path/cus_editing_data.json ./editing_data/
```
Next download the main-backbone and aux-backbone pretrained models at [here](https://pan.baidu.com/s/1Hqq0e4mbYyaMK55UL0oEBQ)(password:fmkx) and put them in ./pretrained_model

Train model with 4 GPUs.

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env tool/train.py
```

## CUS Dataset
CUS Dataset will be released soon.

<img src="https://github.com/zongdai/EditingForDNN/blob/master/image/CUS_images.jpg" width="860"/>

## Concact
For questions regarding our work, feel free to post here or directly contact the authors (zongdai@buaa.edu.cn).
