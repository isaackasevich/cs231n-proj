# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
'''
TODO:
    -Define different loss function 
        -Do we need two loss functions?
    -Define loader_train function for our imagenet data
'''

# from fc import *

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
# from torchvision.datasets import ImageNet

from torchvision.utils import save_image
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

cuda = True if torch.cuda.is_available() else False

# Loss function
adversarial_loss = nn.MSELoss()
# adversarial_loss = nn.BCELoss()


#Initialize FC net
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Expand(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), img_shape[0], img_shape[1], img_shape[2])

input_size = img_shape[0] * img_shape[1] * img_shape[2]
        
fc = nn.Sequential(
    Flatten(),
    nn.Linear(input_size, input_size),
    Expand(),
)

if cuda:
    fc.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer = torch.optim.Adam(fc.parameters(), lr=lr, betas=(b1, b2))

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
dataloader = data.train.loader()
# data = CIFAR10(path, train=True, download=True, 
#                transform=transforms.Compose([
#                    transforms.Lambda(lambda img: PsfBlur_random(img)),
#                    transforms.ToTensor()]))
# dataloader = DataLoader(data, batch_size=5, shuffle=True)

'''
Train

TODO:
    -Experiment with model architectures and params
    -Decide if we still want to add random noise on top of our random blur
        -Assuming we blur randomly
'''

for epoch in range(num_epochs):

    #Loop through batch
    for i, (imgs, _) in enumerate(dataloader):

        #Ground Truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0), requires_grad=False)

        #Configure Input
        real_imgs = Variable(imgs.type(Tensor))

        #Train fc
        optimizer.zero_grad()

        # Sample noise as generator input
        # Not sure we want to do this...if we already have a random filter than we may
        # not need random noise
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images using o
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        loss = adversarial_loss(fc(gen_imgs), valid)

        loss.backward()
        optimizer.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            
            
            
            
            
            
            
 
  
            
            
            
            
            
            
        