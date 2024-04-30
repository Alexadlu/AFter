# AFter
Attention-based Fusion Router for RGBT Tracking.

## Dataset
We use the LasHeR training set for training, GTOT, RGBT210, RGBT234, LasHeR testing set, VTUAVST for testing, and their project addresses are as follows:
* [GTOT](http://chenglongli.cn/code-dataset/)
* [RGBT210](http://chenglongli.cn/code-dataset/)
* [RGBT234](http://chenglongli.cn/code-dataset/)
* [LasHeR](https://github.com/BUGPLEASEOUT/LasHeR)
* [VTUAV](https://github.com/zhang-pengyu/DUT-VTUAV)

## Environment Preparation
Clone repo:  
```
git clone https://github.com/Alexadlu/AFter.git
cd AFter
```
Our code is trained and tested with Python == 3.8, PyTorch == 1.8.1 and CUDA == 11.2 on NVIDIA GeForce RTX 4090, you may use a different version according to your GPU.
```
conda create -n after python=3.8.13
conda activate after
pip install -r requirements.txt
```

## Training

## Evaluation
