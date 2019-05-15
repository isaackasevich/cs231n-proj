## data-loader
## functions that fetch dataset from internet for use in model training

import torch
import torchvision
import torchvision.transforms as transforms
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from PIL import Image

class DataEntry(Dataset):
    def __init__(self, dataset, tsfms = None):
        ## todo: figure out how to get data in the right format
        self.dataset = dataset
        self.transforms = tsfms
     
    def show(idx):
        pass 
    
    def loader(self, batch_size=50, shuffle=True):
        """
        call this function to create and 
        return a data loader for the specified dataset
        
        @param batch_size: batch size to use
        @param shuffle: whether or not to shuffle the dataset
        """
        loader = DataLoader(self.data, 
                            batch_size=batch_size, 
                            shuffle=shuffle)
        return loader
    
    def __len__(self):
        return len(self.data)
    
class CustomCIFAR(CIFAR10):
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        tgt = img.copy()
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            tgt = self.target_transform(tgt)
        
        return img, tgt
        
    
class BlurDataset(object):
    def __init__(self):
        self.train = None
        self.test = None
        
    @staticmethod
    def gopro(path_to_root):
        tsfms = None
        dataset = BlurDataset()
        
        ## ToDo: load in the data files 
#         dataset.train = DataEntry(
        
        return dataset
        
    @staticmethod
    def from_single_dataset(path, dataset_name = 'CIFAR'):
        dataset = BlurDataset()
        if dataset_name == "CIFAR":
            train_data = CustomCIFAR(path+'/train', train=True, 
                                 download = True,
                                 transform = None)
            test_data = CIFAR10(path+'/test', train=False,
                                download = True,
                                transform = None)

#             target = impage.copy()
            
            
#         tsfms = transforms.Compose([
        for t in train_data:
            print(t)
            break
        return dataset
#         dataset.train = DataEntry()
        

"""
Transform classes modified from pytorch tutorial
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

We will write them as callable classes instead of simple functions
so that parameters of the transform need not be passed everytime it’s called.
For this, we just need to implement __call__ method and if required, 
__init__ method. We can then use a transform like this:

tsfm = Transform(params)
transformed_sample = tsfm(sample)
"""

class ChangeTarget(object):
    def __call__(self, sample):
        print(sample)
        img_tgt = sample.copy()
        return {'blurred':img, 'target':img_tgt}
        
class Rescale(object):
    """Rescale the images in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        blurred, target = sample['blurred'], sample['target']
        
        ## blurred and target must have the same shape
        assert blurred.shape == target.shape
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        blurred_resized = transform.resize(blurred, (new_h, new_w))
        target_resized = transform.resize(target, (new_h, new_w))

        return {'blurred': blurred_resized, 'target': target_resized}


class RandomCrop(object):
    """Crop randomly the images in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        blurred, target = sample['blurred'], sample['target']

        ## blurred and target must be same shape
        assert blurred.shape == target.shape
        
        h, w = blurred.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        blurred = blurred[top: top + new_h,
                      left: left + new_w]
        target = target[top: top + new_h,
                      left: left + new_w]

        return {'blurred': blurred, 'target': target}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        blurred, target = sample['blurred'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        blurred = blurred.transpose((2, 0, 1))
        target = target.transpose((2, 0, 1))
        return {'blurred': torch.from_numpy(blurred),
                'target': torch.from_numpy(target)}

class ShiftBlur(object):
    """Blurs the image by applying a shift blur"""
    
    def __init__(self, num_shifts=3):
        self.num_shifts = num_shifts
        
    def __call__(self, sample):
        """
        Want to apply the blur to only the blurred image lol
        """
        blurred, target = sample['blurred'], sample['target']
        
        ## TODO: apply the blur transform (@noah do this)
        
        return {'blurred': blurred,
                'target': target}


    
    
