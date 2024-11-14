# MFMamba
This project is the code of **"MFMamba: A Mamba-Based Multi-Modal Fusion Network for Semantic Segmentation of Remote Sensing Images"** model  
这个项目是  **MFMamba: A Mamba-Based Multi-Modal Fusion Network for Semantic Segmentation of Remote Sensing Images**  的源代码
Our paper's web site: [论文链接](https://doi.org/10.3390/s24227266)
# environment
Our experiments were implemented with the PyTorch framework done on a single NVIDIA A40 GPU equipped with 48GB RAM.  

# Prepare
## dataset 
All datasets including ISPRS Potsdam and ISPRS Vaihingen can be downloaded [here](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)  
首先准备好数据集，数据集下载地址：[ISPRS Potsdam and Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)  
## Pretrained Weights of Backbones 
Download the pre-trained weights in this project:[vmamba_tiny_e292.pth](https://github.com/sstary/SSRS/blob/main/RS3Mamba/pretrain)
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
# Citations
If these codes are helpful for your study, please cite:
```
@Article{s24227266,
AUTHOR = {Wang, Yan and Cao, Li and Deng, He},
TITLE = {MFMamba: A Mamba-Based Multi-Modal Fusion Network for Semantic Segmentation of Remote Sensing Images},
JOURNAL = {Sensors},
VOLUME = {24},
YEAR = {2024},
NUMBER = {22},
ARTICLE-NUMBER = {7266},
URL = {https://www.mdpi.com/1424-8220/24/22/7266},
ISSN = {1424-8220},
ABSTRACT = {Semantic segmentation of remote sensing images is a fundamental task in computer vision, holding substantial relevance in applications such as land cover surveys, environmental protection, and urban building planning. In recent years, multi-modal fusion-based models have garnered considerable attention, exhibiting superior segmentation performance when compared with traditional single-modal techniques. Nonetheless, the majority of these multi-modal models, which rely on Convolutional Neural Networks (CNNs) or Vision Transformers (ViTs) for feature fusion, face limitations in terms of remote modeling capabilities or computational complexity. This paper presents a novel Mamba-based multi-modal fusion network called MFMamba for semantic segmentation of remote sensing images. Specifically, the network employs a dual-branch encoding structure, consisting of a CNN-based main encoder for extracting local features from high-resolution remote sensing images (HRRSIs) and of a Mamba-based auxiliary encoder for capturing global features on its corresponding digital surface model (DSM). To capitalize on the distinct attributes of the multi-modal remote sensing data from both branches, a feature fusion block (FFB) is designed to synergistically enhance and integrate the features extracted from the dual-branch structure at each stage. Extensive experiments on the Vaihingen and the Potsdam datasets have verified the effectiveness and superiority of MFMamba in semantic segmentation of remote sensing images. Compared with state-of-the-art methods, MFMamba achieves higher overall accuracy (OA) and a higher mean F1 score (mF1) and mean intersection over union (mIoU), while maintaining low computational complexity.},
DOI = {10.3390/s24227266}
}
``` 
# Acknowledgement
Many thanks the following projects's contributions to **MFMamba**.  
[RS3Mamba](https://github.com/sstary/SSRS)  
[UNetFormer](https://github.com/WangLibo1995/GeoSeg)  
[PKINet](https://github.com/NUST-Machine-Intelligence-Laboratory/PKINet)  
[SwiftFormer](https://github.com/Amshaker/SwiftFormer)  
