# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
'''
TODO:
    -Define different loss function 
        -Do we need two loss functions?
    -Define loader_train function for our imagenet data
'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
# from torchvision.datasets import ImageNet
from torchvision.datasets import CIFAR10
from pyblur.pyblur.PsfBlur import PsfBlur_random

from torchvision.utils import save_image
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

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

img_shape = (3, 32, 32)
cuda = True if torch.cuda.is_available() else False

# Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Loss function
adversarial_loss = nn.BCELoss()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
#     generator.cuda()
#     discriminator.cuda()
    adversarial_loss.cuda()

# def get_dataloader(path='~/cs231n-proj/data/cifar'):
#     data = CIFAR10(path, train=True, download=True)
#     dataloader = DataLoader(data, batch_size=100, shuffle=True)
#     return dataloader


def train_loop():
    path='~/cs231n-proj/data/cifar'
    data = CIFAR10(path, train=True, download=True, 
                   transform=transforms.Compose([
                       transforms.Lambda(lambda img: PsfBlur_random(img)),
                       transforms.ToTensor()]))
    dataloader = DataLoader(data, batch_size=5, shuffle=True)
    
    num_epochs = 5
    
    for epoch in range(num_epochs):
        
        #Loop through batch
#         for i, (imgs, something) in enumerate(dataloader):        
        for i, (imgs, _) in enumerate(dataloader):

            print("test")
            break
            
    return
      

              
            
#             #Ground Truths
#             valid = Variable(Tensor(imgs.size(0), 1).fill_(1), requires_grad=False)
#             fake = Variable(Tensor(imgs.size(0), 1).fill_(0), requires_grad=False)
        
#             #Configure Input
#             real_imgs = Variable(imgs.type(Tensor))
        
#             #Train generator
#             optimizer_G.zero_grad()
        
#             # Sample noise as generator input
#             # Not sure we want to do this...is this where we apply our filter?
#             # Update: I think we do want random noise but also our filter? Or
#             # incorporate randomness into our filter
#             z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

#             # Generate a batch of images using o
#             gen_imgs = generator(z)

#             # Loss measures generator's ability to fool the discriminator
#             g_loss = adversarial_loss(discriminator(gen_imgs), valid)

#             g_loss.backward()
#             optimizer_G.step()
            
            
            
#             #Train Discriminator
#             optimizer_D.zero_grad()

#             # Measure discriminator's ability to classify real from generated samples
#             real_loss = adversarial_loss(discriminator(real_imgs), valid)
#             fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
#             d_loss = (real_loss + fake_loss) / 2

#             d_loss.backward()
#             optimizer_D.step()
            
#              print(
#             "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#             % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
#             )

#             batches_done = epoch * len(dataloader) + i
#             if batches_done % opt.sample_interval == 0:
#                 save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            
            
            
            
            
            
            
 
  
            
            
            
            
            
            
        