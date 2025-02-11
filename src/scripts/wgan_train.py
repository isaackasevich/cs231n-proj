import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from torchvision.utils import save_image
from torch.autograd import Variable

import sys
sys.path.append('../dataprocessing')
sys.path.append('../models')
from dataloader import BlurDataset

import torch.nn as nn
import torch.nn.functional as F

from wgan import WGANGenerator as Generator
from wgan import WGANDiscriminator as Discriminator
from wgan import generator_loss, discriminator_loss, FeatureExtractor
from metrics import reconstruction_loss, append_to_file

'''
Initialization 
'''
num_epochs = 1
batch_size = 25
# noise_dim = 96
channels = 3
size = 200 
img_shape = (channels, size, size)
lr = 1e-4
b1 = .5
b2 = .999
sample_interval = 50
save_interval = 10
validation_rate = 1   #Test on validation set once an epoch

input_size = img_shape[0] * img_shape[1] * img_shape[2]
   
if torch.cuda.is_available(): dtype = torch.cuda.FloatTensor
else: dtype = torch.FloatTensor

#Models
file_base = "../../outputs/wgan_content_loss/"
load = True
# load = False

device_1 = torch.device("cuda:0")
device_2 = torch.device("cuda:1")
device_ids = [0]

if load:
    generator = torch.load(file_base + "generator_2loss.pt")
    discriminator = torch.load(file_base + "discriminator_2loss.pt")
else:
    generator = Generator(img_shape)
    discriminator = Discriminator(img_shape)


generator = nn.DataParallel(generator, device_ids=device_ids).type(dtype)
discriminator = nn.DataParallel(discriminator, device_ids=device_ids).type(dtype)
    
feature_extractor = FeatureExtractor().type(dtype)
# Optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

'''
Load Data 
'''
    
'''
Train

TODO:
    -Experiment with params
'''
def sample_noise(bs, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.
    bs: batch size
    dim: noise dim
    """
    return torch.rand(bs, dim)*2 - 1.0    

def save_losses(losses, path):
    with open(path, 'w') as f:
        for item in losses:
            f.write("{}\n".format(item))

def train_wgan_batches(epoch, G, D, dataloader, batch_size, it=0, train=True, save=True):
    losses = []
    #Loop through batch
    for i, (imgs, tgts) in enumerate(dataloader):

        #Send imgs to gpu
        imgs = imgs.type(dtype)
        tgts = tgts.type(dtype)

        ### Train generator ###
        g_optimizer.zero_grad()
        
        # Generate images using the generator, find loss
#         noise = sample_noise(batch_size, noise_dim).type(dtype)
        gen_imgs = G(imgs)
        g_loss = generator_loss(gen_imgs, tgts, D, feature_extractor)
        if train:
            g_loss.backward()
            g_optimizer.step()

        ### Train discriminator ###
        d_optimizer.zero_grad()
        
        d_loss = discriminator_loss(D(tgts), D(gen_imgs.detach()))
        if train:
            d_loss.backward()
            d_optimizer.step()
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

        iteration = it + epoch * len(dataloader) + i 
        pathval = "train" if train else "val"
        
        r_loss = F.mse_loss(gen_imgs, tgts)

        print(pathval + " [Epoch %d/%d] [Batch %d/%d] [Iter %d] [G_Loss %f] [D_Loss %f] [R_Loss %g]"
        % (epoch, num_epochs, i, len(dataloader), iteration, g_loss, d_loss, r_loss))
        
#         if iteration % sample_interval == 0:
#             losses.append((r_loss.item(), g_loss.item(), d_loss.item(), iteration))
        for j in range(25):
            save_image(imgs.data[j], file_base + "%d_" % iteration + pathval + "_input_%d.png" % j, nrow=1)
            save_image(gen_imgs.data[j], file_base + "%d_" % iteration + pathval + "_output_%d.png" % j, nrow=1)
            save_image(tgts.data[j], file_base + "%d_" % iteration + pathval + "_target_%d.png" % j, nrow=1)
        break
        if save and (iteration % save_interval == 0): 
            metric = reconstruction_loss(gen_imgs, tgts)
            append_to_file("%i, %.5f\n"%(iteration, metric), file_base + "reconstruction_losses_" + pathval + ".txt")
            torch.save(generator, file_base + "generator_2loss.pt")
            torch.save(discriminator, file_base + "discriminator_2loss.pt")
            
    return losses

def train_wgan(generator, discriminator, save=True, it=0):
    data_path='../../data/coco'
#     batch_size = 16
    data = BlurDataset.from_single_dataset(data_path)
    train_dataloader = data.loader(batch_size=batch_size)
    val_dataloader = data.loader(split='test', batch_size=batch_size)

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
#         train_losses += train_wgan_batches(epoch, generator, discriminator, train_dataloader, batch_size, it)
        
#         if epoch % validation_rate == 0:
        val_losses += train_wgan_batches(epoch, generator, discriminator, val_dataloader, batch_size, it=0, train=False, save=False)
#     save_losses(train_losses, file_base + "train_lossses.txt")
#     save_losses(val_losses, file_base + "val_losses.txt")
            
            
