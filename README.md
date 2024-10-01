# MFMamba
This project is the code of **MFMamba** model  
这个项目是MFMamba的源代码
# environment
Our experiments were implemented with the PyTorch framework done on a single NVIDIA A40 GPU equipped with 48GB RAM.  

# Prepare
## dataset 
All datasets including ISPRS Potsdam and ISPRS Vaihingen can be downloaded [here](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)  
首先准备好数据集，数据集下载地址：[ISPRS Potsdam and Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)  
## Pretrained Weights of Backbones 

## Folder Structure
```
MFMamba
├── pretrain (pretrained weights of backbones)
├── model (models)
├── train_MFMamba.py (Training code)
├── utils_Mamba.py (Configuration)
├── results (The folder to save the results)
├── data
│   ├── vaihingen
│   │   ├── dsm (original)
│   │   ├── top (original)
│   │   ├── gts_eroded_for_participants (original)
│   │   ├── gts_for_participants (original)
│   ├── potsdam (the same with vaihingen)
```
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
### Modify the parameters and addresses
1. Modify the address of data in utils_Mamba.py, batch_size, training mode =train, training model =MFMamba.  
（修改utils_Mamba.py中的data的地址、batch_size、训练模式=train、训练模型=MFMamba）  
2. Modify the address of results in train_MFMamba.py
### train
```
python train_MFMamba.py
``` 
# test
### Modify the parameters and addresses
1. Modify the address of results in train_MFMamba.py to be the address of the best trained model  
2. Modify the training mode =test in utils_Mamba.py
### test
```
python train_MFMamba.py
```
# Contact
Yan Wang（w2268388154@163.com）
# Acknowledgement
Many thanks the following projects's contributions to **MFMamba**.  
[RS3Mamba](https://github.com/sstary/SSRS)  
[UNetFormer](https://github.com/WangLibo1995/GeoSeg)  
[PKINet](https://github.com/NUST-Machine-Intelligence-Laboratory/PKINet)  
[SwiftFormer](https://github.com/Amshaker/SwiftFormer)  
