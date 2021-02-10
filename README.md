# 3D Part Guided Image Editing for Fine-grained Object Understanding(CVPR2020)
[CVPR 2020 Paper](https://drive.google.com/file/d/1dX7yD-qhF7HPSumeCajOjqfmM8_er129/view?usp=sharing) | [Video](https://drive.google.com/file/d/1PSnZCbsN7sMXQ9tmKf2dmqn3nVRzHDXh/view?usp=sharing)


Zongdai Liu, Feixiang Lu, Peng Wang, Hui Miao, Liangjun Zhang, Ruigang Yang, Bin Zhou
## Abstrct
Holistically understanding an object with its 3D movable parts is essential for visual models of a robot to interact with the world. For example, only by understanding many possible part dynamics of other vehicles (e.g., door or trunk opening, taillight blinking for changing lane), a self-driving vehicle can be success in dealing with emergency cases. However, existing visual models tackle rarely on these situations, but focus on bounding box detection. In this paper, we fill this important missing piece in autonomous driving by solving two critical issues. First, for dealing with data scarcity, we propose an effective training data generation process by fitting a 3D car model with dynamic parts to cars in real images. This allows us to directly edit the real images using the aligned 3D parts, yielding effective training data for learning robust deep neural networks (DNNs). Secondly, to benchmark the quality of 3D part understanding, we collected a large dataset in real driving scenario with cars in uncommon states (CUS), i.e. with door or trunk opened etc., which demonstrates that our trained network with edited images largely outperforms other baselines in terms of 2D detection and instance segmentation accuracy.


<img src="https://github.com/zongdai/EditingForDNN/blob/master/image/Overview.jpg" width="860"/>



## Data
The editing data totally 27k could be downloaded at [BaiduNetdisk](https://pan.baidu.com/s/1UW6VmnYbeuvnxJm9rvuKZw)(password:kmve)

