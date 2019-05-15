## data-loader
## functions that fetch dataset from internet for use in model training

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
# from torchvision.datasets import CIFAR

def get_dataset(path='~/cs231n-proj/data/imagenet'):
    data = ImageNet(path, train=True, download=True)
    dataloader = DataLoader(data, batch_size=100, shuffle=True)
    return dataloader
    


# data_train = ImageNet('~/cs231n-proj/data', split='train', download=True)
# data_val = ImageNet('~/cs231n-proj/data', split='val', download=True)

# def get_dataset(store_path = '../data/imagenet'):
#     '''
#     This function loads a pytorch dataset and stores it in a the provided directory
#     '''
#     store_path += dataset_name
#     train_data = CIFAR(store_path, split='train', download=False)
#     test_data = CIFAR(store_path, split='val', download=False)
#     return train_data, test_data
    
    
    
