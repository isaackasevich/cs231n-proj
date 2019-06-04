## data-loader
## functions that fetch dataset from internet for use in model training
import sys, os

import torch
import torchvision
import torchvision.transforms as transforms
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import scipy
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CocoDetection
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from pyblur.pyblur.LinearMotionBlur import LinearMotionBlur, randomAngle

from pyblur.pyblur.PsfBlur import PsfBlur
from PIL import Image
import random
from IPython.display import display

class BlurDataset(object):
    """
    Generic dataset for this ML task
    """
    def __init__(self, data, val_split = 0.025, test_split = 0.025, shuffle=True):
        ## create splits
        self.data = data
        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None
        self._create_splits(val_split, test_split, shuffle)
        
    def _create_splits(self, val = 0.025, test = 0.025, shuffle=True):
        # Creating data indices for training and validation splits:
        dataset_size = len(self.data)
        indices = list(range(dataset_size))
        first_split = int(np.floor(test * dataset_size))
        second_split = int(np.floor((val+val) * dataset_size))
        if shuffle :
            np.random.shuffle(indices)
            
        test_inds = indices[:first_split]
        val_inds = indices[first_split:second_split]
        train_inds = indices[second_split:]
        
        # Creating data samplers, setting them to class variables:
        self.train_sampler = SubsetRandomSampler(train_inds)
        self.val_sampler = SequentialSampler(val_inds)
        self.test_sampler = SequentialSampler(test_inds)
   
    def loader(self, split = 'train', batch_size=50):
        """
        call this function to create and 
        return a data loader for the specified dataset
        
        @param train: whether to load training data or test data
        @param batch_size: batch size to use
        """
        sampler = None
        if split == 'train':
            sampler = self.train_sampler
        elif split == 'val':
            sampler = self.val_sampler
        elif split == 'test':
            sampler = self.test_sampler
        
        assert sampler is not None, "Incorrect split specified"
        
        loader = DataLoader(self.data, 
                            batch_size = batch_size,
                            sampler = sampler,
                            shuffle = False,
                            drop_last = True)

        return loader
    
    
    @staticmethod
    def gopro(path_to_root):
        tsfms = None
        dataset = BlurDataset()
        
        ## ToDo: load in the data files 
#         dataset.train = DataEntry(
        return dataset
        
    @staticmethod
    def from_single_dataset(path, 
                            dataset_name = 'coco', 
                            val_split = 0.025, 
                            test_split = 0.025):
        
        if dataset_name == "cifar":
            ## cifar 10 small dataset to test stuff
            data = CustomCIFAR(path+'/train', train = True, 
                                 download = True,
                                 transform = transforms.ToTensor(),
                                 target_transform = Blur('linear',
                                                        line_lengths = [3,5]))
            
        elif dataset_name == "coco":
            ## Microsoft common objects in context datset
            data = CustomCoco(path +'/images', path+'/annotations.json', 
                              transform = transforms.Compose([
                                  transforms.Resize(200),
                                  transforms.CenterCrop(200),
                                  transforms.ToTensor()
                              ]),
                              target_transform = Blur('linear', 
                                                 line_lengths = [9]))
                                                       
        dataset = BlurDataset(data, val_split, test_split)
        
        return dataset
    
class CustomCIFAR(CIFAR10):
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        tgt = img.copy()
        
        if self.target_transform is not None:
            img = self.target_transform(img)
        
        ## caution! only use deterministic transforms here
        if self.transform is not None:
            img = self.transform(img)
            tgt = self.transform(tgt)

        
        return img, tgt
        
class CustomCoco(CocoDetection):
    """
    transform: applied to both input and target
    target_transform: wrong name (should be named blur_transform) but dont want
        to override too much; this transform will only be applied to the ***input***
        image (think of it as an un-blur for the target if you like)
    
    """
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        target = img.copy()
        
        if self.target_transform is not None:
            ## apply blur (e.g. 'target_transform')
            img = self.target_transform(img)
         
        ## caution! only use deterministic transforms or write your own
        ## that take in multiple images
        if self.transform is not None:
            ## transform both images
            img = self.transform(img)
            target = self.transform(target)

        return img, target
        
        

        

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


######## These are only for gopro dataset which has diff format #########        
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

class Blur(object):
    """Blurs the image by applying a blur"""
    
    def __init__(self, blur_kernel = 'psf', **kwargs):
        self.blur_kernel = blur_kernel
        self.line_lengths = kwargs.get('line_lengths', [3,5,7,9])
        
    def __call__(self, image):
        """
        Want to apply the blur to only the blurred image lol
        """
        image = np.asarray(image)
        img_r = image[:,:,0]
        img_g = image[:,:,1]
        img_b = image[:,:,2]
        if self.blur_kernel == 'psf': 
            ## Point-spread function blur kernel
            psfid = np.random.randint(0, 99)
            blurred_r = PsfBlur(img_r, psfid)
            blurred_g = PsfBlur(img_g, psfid)
            blurred_b = PsfBlur(img_b, psfid)
            
        elif self.blur_kernel == 'linear':
            ## Linear Motion blur kernel
            dim = self.line_lengths[np.random.randint(0, len(self.line_lengths))]
           
            angle = randomAngle(dim)
            line_type = 'full'
            blurred_r = LinearMotionBlur(img_r, dim, angle, line_type)
            blurred_g = LinearMotionBlur(img_g, dim, angle, line_type)
            blurred_b = LinearMotionBlur(img_b, dim, angle, line_type)
        else:
            ## no blur, just replace image normally
            blurred_r = img_r
            blurred_g = img_g
            blurred_b = img_b
            
        blurred_image = np.zeros_like(image)
        blurred_image[:,:,0] = blurred_r
        blurred_image[:,:,1] = blurred_g
        blurred_image[:,:,2] = blurred_b
        blurred_image = Image.fromarray(blurred_image)
        
        return blurred_image

