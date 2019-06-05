# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
'''
TODO:
    -Define different loss function 
        -Do we need two loss functions?
    -Define loader_train function for our imagenet data
'''


import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from torchvision.utils import save_image
from torch.autograd import Variable

import sys
sys.path.append('../dataprocessing')
from dataloader import BlurDataset

import torch.nn as nn
import torch.nn.functional as F

from gan import *

'''
Initialization 

TODO: 
    -Decide on what we want our args to be
    -Make sure img_shape is right
'''
num_epochs = 2
batch_size = 100
channels = 3
size = 200 
img_shape = (channels, size, size)
lr = 1e-3
b1 = .5
b2 = .999
sample_interval = 25
save_interval = 25

if torch.cuda.is_available(): dtype = torch.cuda.FloatTensor
else: dtype = torch.FloatTensor

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Expand(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), img_shape[0], img_shape[1], img_shape[2])

input_size = img_shape[0] * img_shape[1] * img_shape[2]

def conv_net():
    model = nn.Sequential(
        nn.Conv2d(3, 7, 5, padding=2),
        nn.ReLU(),
        nn.Conv2d(7, 5, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(5, 3, 3, padding=1),
    )
    return model

def content_loss(feature_extractor, gen_imgs, tgt_imgs):
    loss = F.mse_loss(feature_extractor(gen_imgs), feature_extractor(tgt_imgs))
    return loss

#Model
conv_model = conv_net().type(dtype)

# Loss function
loss_func = nn.MSELoss().type(dtype)

# Optimizers
optimizer = torch.optim.Adam(conv_model.parameters(), lr=lr, betas=(b1, b2))

'''
Load Data 

TODO:
    -Apply a transform to incoming images
    -Split to train and val
    -
'''
    
data_path='../../data/coco'

data = BlurDataset.from_single_dataset(data_path)
train_dataloader = data.loader(batch_size=batch_size)
val_dataloader = data.loader(split='val', batch_size=batch_size)
'''
Train

TODO:
    -Experiment with model architectures and params
    -Decide if we still want to add random noise on top of our random blur
        -Assuming we blur randomly
'''
def save_losses(losses, path):
    with open(path, 'a+') as f:
        for item in losses:
            f.write("{}\n".format(item))

def train_batches(epoch, conv_model, dataloader, it=0, train=True, save=True):
    losses = []
    #Loop through batch
    for i, (imgs, tgts) in enumerate(dataloader):

        #Send imgs to gpu
        imgs = imgs.type(dtype)
        tgts = tgts.type(dtype)

        ### Train conv ###
        optimizer.zero_grad()

        # Generate images using the generator, find loss
        gen_imgs = conv_model(imgs)
        loss = loss_func(gen_imgs, tgts)
        if train:
            loss.backward()
            optimizer.step()
            
        iteration = it + epoch * len(dataloader) + i
        pathval = "train" if train else "val"

        print(pathval + " [Epoch %d/%d] [Batch %d/%d] [Iter %d] [Loss %f] "
        % (epoch, num_epochs, i, len(dataloader), iteration, loss))

        if iteration % sample_interval == 0 or not train:
            cur_loss = [(iteration, loss.item())]
            losses += cur_loss
            save_losses(cur_loss, "../../outputs/conv/" + pathval + "_losses.txt")
            save_image(imgs.data[:4], "../../outputs/conv/%d_" % iteration + pathval + "_input.png", nrow=2)
            save_image(gen_imgs.data[:4], "../../outputs/conv/%d_" % iteration + pathval + "_output.png", nrow=2)
            save_image(tgts.data[:4], "../../outputs/conv/%d_" % iteration + pathval + "_target.png", nrow=2)
        if save and (iteration % save_interval == 0): 
            torch.save(conv_model, "../../outputs/conv/conv_model.pt")

def train_model(conv_model, save=True, it=0):
    
    for epoch in range(num_epochs):
        train_batches(epoch, conv_model, train_dataloader, it, save=save)

        if epoch % validation_rate == 0:
            train_batches(epoch, conv_model, val_dataloader, it, train=False, save=False)

    
            
            
            
            
            
 
  
            
            
            
            
            
            
        