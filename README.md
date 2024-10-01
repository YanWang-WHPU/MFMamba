# MFMamba
This project is the code of MFMamba model  
这个项目是MFMamba的源代码
# environment
Our experiments were implemented with the PyTorch framework done on a single NVIDIA A40 GPU equipped with 48GB RAM.  

# Prepare
## dataset 
All datasets including ISPRS Potsdam and ISPRS Vaihingen can be downloaded [here](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)  
首先准备好数据集，数据集下载地址：[ISPRS Potsdam and Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)  
## Pretrained Weights of Backbones 

训练权重下载
## requirements

# Install
### Open the folder MFMamba using Linux Terminal and create python environment(创建环境):
```
conda create -n MFMamba python=3.8 -y
conda activate MFMamba
```
### Install cuda=11.8(已安装可忽略)
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
```
### Install torch=2.0.0(已安装可忽略）
```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
or(或者pip安装):
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 
```
### Install Mamba
#### Install mamba_ssm
```
pip install mamba-ssm
```
Because installing with pip can be problematic, we recommend downloading and installing it [here](https://github.com/state-spaces/mamba/releases)
#### Install causal-conv1d
```
pip install causal-conv1d
```
Because installing with pip can be problematic, we recommend downloading and installing it [here](https://github.com/Dao-AILab/causal-conv1d/releases)
# Train  
```
python train_MFMamba.py
``` 
