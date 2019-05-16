# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
'''
TODO:
    -Define different loss function 
        -Do we need two loss functions?
    -Define loader_train function for our imagenet data
'''

import numpy as np

from gan import *

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from torchvision.utils import save_image, make_grid
from torch.autograd import Variable

import sys
sys.path.append('../dataprocessing')
from dataloader import BlurDataset

import torch.nn as nn
import torch.nn.functional as F

# parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
# opt = parser.parse_args()
# print(opt)

'''
Initialization 

TODO: 
    -Decide on what we want our args to be
    -Make sure img_shape is right
'''
num_epochs = 5
channels = 3
size = 32    #May need to change for earlier tests
img_shape = (3, 32, 32)
lr = .0002
b1 = .5
b2 = .999
sample_interval = 50

cuda = True if torch.cuda.is_available() else False
if cuda: device = torch.device('cuda')
else: device = torch.device('cpu')

#Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(3, 32, 32))
feature_extractor = FeatureExtractor()

# Loss function
criterion_GAN = nn.MSELoss()
criterion_content = nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
#     criterion_content = criterion_content.cuda()
    
#Load pretrained models
# generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
# discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

'''
Load Data 

TODO:
    -Apply a transform to incoming images
    -Split to train and val
    -
'''
    
path='~/cs231n-proj/data/cifar'

data = BlurDataset.from_single_dataset(path)
dataloader = data.train.loader(batch_size=5)

'''
Train

TODO:
    -Experiment with model architectures and params
    -Decide if we still want to add random noise on top of our random blur
        -Assuming we blur randomly
'''

def train_model():

    for epoch in range(num_epochs):

        #Loop through batch
        for i, (imgs, tgts) in enumerate(dataloader):

            print(*discriminator.output_shape)
            #Ground Truths
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs.size(0), 1,8,8))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs.size(0), *discriminator.output_shape))), requires_grad=False)

            #Configure Input
            imgs = Variable(imgs.type(Tensor))
            tgts = Variable(tgts.type(Tensor)).to(device)

            #Train generator
            optimizer_G.zero_grad()

            # Generate a batch of images using o
            gen_imgs = generator(imgs)

            # Adversarial loss
            gan_loss = criterion_GAN(discriminator(gen_imgs), valid)
            
            # Content loss
            gen_features = feature_extractor(gen_imgs)
            real_features = feature_extractor(tgts)
#             loss_content = criterion_content(gen_features, real_features.detatch())

            g_loss = 1e-3 * gan_loss
            
            g_loss.backward()
            optimizer_G.step()

            #Train Discriminator
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            print(tgts.shape, gen_imgs.shape, imgs.shape)
            loss_real = criterion_GAN(discriminator(tgts), valid)
            loss_fake = criterion_GAN(discriminator(gen_imgs), fake)
            d_loss = (loss_real + loss_fake)/2

            d_loss.backward()
            optimizer_D.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss_D.item(), g_loss.item()))

#             batches_done = epoch * len(dataloader) + i
#             if batches_done % opt.sample_interval == 0:
#                 save_image(gen_imgs.data[:25], "../../outputs/gan_out%d.png" % batches_done, nrow=5, normalize=True)
#                 save_image(tgts.data[:25], "../../outputs/in.png", nrow=5)
            
            
            
            
            
            
            
 
  
            
            
            
            
            
            
        