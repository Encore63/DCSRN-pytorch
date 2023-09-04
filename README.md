# DCSRN-pytorch-master

## Introduction
3D deep densely connected neural network (DCSRN) implemented using PyTorch

## Prerequisites
>> pytorch >= 1.7.1  
>> numpy == 1.21.5  
>> opencv-python == 4.5.3.56  
>> tensorboard == 2.5.0  
>> torchinfo == 1.8.0  
>> tqdm == 4.64.0  
>> torchvision == 0.2.2  
 
## Usage
- Run data_prepare.py to spilt the raw dataset into train dataset, eval dataset and test dataset, than merge the .nii format files and finally generate the .npy files.
- Run demo.py to test the effect of pre-trained DCSRN model on single MRI picture.
- Run train.py, test.py to train and test DCSRN model respectively.
