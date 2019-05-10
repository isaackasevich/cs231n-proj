## data-loader
## functions that fetch dataset from internet for use in model training

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torchvision.datasets import CIFAR

def get_dataset(store_path = '../data/imagenet'):
    '''
    This function loads a pytorch dataset and stores it in a the provided directory
    '''
    store_path += dataset_name
    train_data = CIFAR(store_path, split='train', download=False)
    test_data = CIFAR(store_path, split='val', download=False)
    return train_data, test_data
    
    
    
