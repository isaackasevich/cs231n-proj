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
'''
num_epochs = 1
batch_size = 50
channels = 3
size = 200 
img_shape = (channels, size, size)
lr = 2e-4
b1 = .5
b2 = .999
sample_interval = 25
save_interval = 25
validation_rate = 1   #Test on validation set once an epoch

input_size = img_shape[0] * img_shape[1] * img_shape[2]
   
if torch.cuda.is_available(): dtype = torch.cuda.FloatTensor
else: dtype = torch.FloatTensor
    
pixel_loss = torch.nn.MSELoss().type(dtype)

#Models
generator = Generator(img_shape).type(dtype)
discriminator = Discriminator(img_shape).type(dtype)
feature_extractor = FeatureExtractor().type(dtype)

# Optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

'''
Load Data 
'''
    
data_path='../../data/coco'

data = BlurDataset.from_single_dataset(data_path)
train_dataloader = data.loader(batch_size=batch_size)
val_dataloader = data.loader(split='val', batch_size=batch_size)
'''
Train

TODO:
    -Experiment with params
'''
def save_losses(losses, path):
    with open(path, 'a+') as f:
        for item in losses:
            f.write("{}\n".format(item))

def train_batches(epoch, generator, discriminator, dataloader, it=0, train=True, save=True):
    losses = []
    #Loop through batch
    for i, (imgs, tgts) in enumerate(dataloader):

        #Send imgs to gpu
        imgs = imgs.type(dtype)
        tgts = tgts.type(dtype)

        ### Train generator ###
        g_optimizer.zero_grad()

        # Generate images using the generator, find loss
        gen_imgs = generator(imgs)
        g_loss = generator_loss(discriminator(gen_imgs), feature_extractor, gen_imgs, tgts)
        if train:
            g_loss.backward()
            g_optimizer.step()

        ### Train discriminator ###
        d_optimizer.zero_grad()

        d_loss = discriminator_loss(discriminator(tgts), discriminator(gen_imgs.detach()))
        if train:
            d_loss.backward()
            d_optimizer.step()

        iteration = it + epoch * len(dataloader) + i
        pathval = "train" if train else "val"
        
        p_loss = pixel_loss(gen_imgs, tgts)
        
        print(pathval + " [Epoch %d/%d] [Batch %d/%d] [Iter %d] [G_Loss %f] [D_Loss %f] [P_Loss %f] "
        % (epoch, num_epochs, i, len(dataloader), iteration, g_loss, d_loss, p_loss))

        if iteration % sample_interval == 0:
            cur_loss = [(iteration, p_loss.item(), g_loss.item(), d_loss.item())]
            losses += cur_loss
            save_losses(cur_loss, "../../outputs/gan/" + pathval + "_losses.txt")
            save_image(imgs.data[:4], "../../outputs/gan/%d_" % iteration + pathval + "_input.png", nrow=2)
            save_image(gen_imgs.data[:4], "../../outputs/gan/%d_" % iteration + pathval + "_output.png", nrow=2)
            save_image(tgts.data[:4], "../../outputs/gan/%d_" % iteration + pathval + "_target.png", nrow=2)
        if save and (iteration % save_interval == 0): 
            torch.save(generator, "../../outputs/gan/generator.pt")
            torch.save(discriminator, "../../outputs/gan/discriminator.pt")

def train_model(generator, discriminator, save=True, it=0):
    
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
#         train_batches(epoch, generator, discriminator, train_dataloader, it)

        if epoch % validation_rate == 0:
            train_batches(epoch, generator, discriminator, val_dataloader, it, train=False, save=False)
            
            
            
            
 
  
            
            
            
            
            
            
        