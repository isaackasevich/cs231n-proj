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

'''
Define UNet classes
'''
 
class UNet(nn.Module):
    def __init__(self, img_shape, channel_1=16, channel_2=12, channel_3=20, channel_4=12):
        super().__init__()
        
        in_channel, H, W = img_shape

        self.conv1 = nn.Conv2d(in_channel, channel_1, 3, padding=1)    
        self.conv2 = nn.Conv2d(channel_1, channel_2, 3, padding=1)
        self.conv3 = nn.Conv2d(channel_2, channel_3, 3, padding=1)
        
        self.unconv1 = nn.ConvTranspose2d(channel_3, channel_2, 4, stride=2, padding=1)
        self.unconv2 = nn.ConvTranspose2d(2*channel_2, channel_1, 4, stride=2, padding=1)
        
        self.conv4 = nn.Conv2d(2*channel_1, channel_4, 3, padding=1)
        self.out_conv = nn.Conv2d(channel_4, 3, 3, padding=1)
        
        self.maxpool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.out_conv.weight)
        nn.init.kaiming_normal_(self.unconv1.weight)
        nn.init.kaiming_normal_(self.unconv2.weight)
        
    def forward(self, x):
        o1 = self.relu(self.conv1(x))
        m1 = self.maxpool(o1)
        o2 = self.relu(self.conv2(m1))
        m2 = self.maxpool(o2)
        o3 = self.relu(self.conv3(m2))
        
        t4 = self.relu(self.unconv1(o3))
        o4 = torch.cat((t4, o2), dim=1)
        
        t5 = self.relu(self.unconv2(o4))
        o5 = torch.cat((t5, o1), dim=1)
        
        o6 = self.relu(self.conv4(o5))
        out_img = self.relu(self.out_conv(o6))
        
        return out_img
        
'''
Initialization 
'''
num_epochs = 1
batch_size = 100
channels = 3
size = 64 
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
    
#Model
unet = UNet(img_shape).type(dtype)

#Loss
loss_func = torch.nn.MSELoss().type(dtype)

# Optimizers
optimizer = torch.optim.Adam(unet.parameters(), lr=lr, betas=(b1, b2))

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

def train_batches(epoch, unet, dataloader, it=0, train=True, save=True):
    losses = []
    #Loop through batch
    for i, (imgs, tgts) in enumerate(dataloader):

        #Send imgs to gpu
        imgs = imgs.type(dtype)
        tgts = tgts.type(dtype)

        ### Train UNet ###
        optimizer.zero_grad()

        # Generate images using the generator, find loss
        gen_imgs = unet(imgs)
        loss = loss_func(gen_imgs, tgts)
        if train:
            loss.backward()
            optimizer.step()

        ### Train discriminator ###
        optimizer.zero_grad()

        iteration = it + epoch * len(dataloader) + i
        pathval = "train" if train else "val"
        
        print(pathval + " [Epoch %d/%d] [Batch %d/%d] [Iter %d] [Loss %f] "
        % (epoch, num_epochs, i, len(dataloader), iteration, loss))

        if iteration % sample_interval == 0:
            cur_loss = [(iteration, loss.item())]
            losses += cur_loss
            save_losses(cur_loss, "../../outputs/unet/" + pathval + "_losses.txt")
            save_image(imgs.data[:4], "../../outputs/unet/%d_" % iteration + pathval + "_input.png", nrow=2)
            save_image(gen_imgs.data[:4], "../../outputs/unet/%d_" % iteration + pathval + "_output.png", nrow=2)
            save_image(tgts.data[:4], "../../outputs/unet/%d_" % iteration + pathval + "_target.png", nrow=2)
        if save and (iteration % save_interval == 0): 
            torch.save(unet, "../../outputs/unet/unet.pt")

def train_model(unet, save=True, it=0):
    
    for epoch in range(num_epochs):
        train_batches(epoch, unet, train_dataloader, it)

        if epoch % validation_rate == 0:
            train_batches(epoch, unet, val_dataloader, it, train=False, save=False)
            
            
            
            
 
  
            
            
            
            
            
            
        