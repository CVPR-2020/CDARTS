# Cyclic Differential Architecture Search: A Unified Architecture for Integrating Search and Evaluation Networks
In this paper, motivated by the separation problem of the search and evaluation networks in one-shot DARTS, we proposed a cyclic differential architecture search framework which integrates the two networks into a unified architecture. The alternating joint learning enables the search of cell architectures to fit the final evaluation network. Experiments demonstrate the effectiveness of the proposed algorithm and searched architectures, leading to competitive performance on both CIFAR and ImageNet. Our model achieves competitive performance in comparison with the state-of-the-art methods on three benchmark datasets.
<div align="center">
  <img src="images/framework1.png" width="350px" />
  <!-- <p>cell.</p> -->
</div>

## Features of CDARTS
- :star2: We achieved the best or comparable performance on CIFAR10(97.60%), CIFAR100(84.31%) and ImageNet (75.9%) under the same DARTS searching space!
- :star2: Our big model achieved impressive performance on CIFAR10(98.32%), CIFAR100(87.01%) and ImageNet (80.2%)!
- :star2: Our code support distributed training.

## Main Results
<div align="center">
  <img src="images/cell.png" width="800px" />
  <!-- <p>cell.</p> -->
</div>

<div align="center">
  <img src="images/results1.png" width="800px" />
  <!-- <p>results1.</p> -->
</div>

<div align="center">
  <img src="images/results2.png" width="800px" />
  <!-- <p>results1.</p> -->
</div>

## Prerequisites
* CUDA 10.0
* pytorch >= 1.2
* python >= 3
* [apex](https://github.com/NVIDIA/apex)

## Quick Start

#### Data Preparation
please download this 3 datasets following official steps.

* [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [Cifar-100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [ImageNet-2012](http://www.image-net.org/)

Then create soft link in main dir.
```
ln -s $DataLocation data
```

#### Installation
First, you should install graphviz.  
```
sudo apt-get install graphviz
```  
Install python requirements.  
```buildoutcfg
pip install graphviz
pip install torch==1.2.0
pip install tensorboard==1.13.0
pip install tensorboardX==1.6
```  
Then install apex.  
```buildoutcfg
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext
```  

#### Testing
Main python file is ${ROOT}/test.py. Followings are options during testing.
```buildoutcfg
--resume                   # whether to load checkpint
--resume_name              # checkpint name
```  

Our CIFAR10, CIFAR100 and ImageNet models are in the following paths.
```buildoutcfg
${CODE_ROOT}/augments/cifar10-retrain/c10.pth.tar               # CIFAR10 Checkpoint
${CODE_ROOT}/augments/cifar100-retrain/c100.pth.tar             # CIFAR100 Checkpoint
${CODE_ROOT}/augments/imagenet-retrain/imagenet.pth.tar         # ImageNet Checkpoint
```  

Here we present our test scripts on CIFAR10, CIFAR100 and ImageNet.  
```buildoutcfg
cd ${CODE_ROOT}/
bash run_test_cifar10.sh
bash run_test_cifar100.sh
bash run_test_imagenet.sh
```
**We will release the training source code when the paper is accepted.**
