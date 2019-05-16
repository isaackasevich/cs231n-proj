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
batch_size = 100
channels = 3
size = 200 
img_shape = (channels, size, size)
lr = 1e-3
b1 = .5
b2 = .999
sample_interval = 50

cuda = True if torch.cuda.is_available() else False
print(cuda)
if cuda: device = torch.device('cuda')
else: device = torch.device('cpu')

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Expand(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), img_shape[0], img_shape[1], img_shape[2])

input_size = img_shape[0] * img_shape[1] * img_shape[2]
   
def fc_net():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(input_size, input_size),
        Expand(),
    )
    return model

def conv_net():
    model = nn.Sequential(
        nn.Conv2d(3, 7, 5, padding=2),
        nn.ReLU(),
        nn.Conv2d(7, 5, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(5, 3, 3, padding=1),
    )
    return model

model = conv_net()

# Loss function
loss_func = nn.MSELoss()
    
if cuda:
    model = model.to(device=device)
    loss_func = loss_func.to(device=device)

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

'''
Load Data 

TODO:
    -Apply a transform to incoming images
    -Split to train and val
    -
'''
    
path='../../data/coco'

data = BlurDataset.from_single_dataset(path)
dataloader = data.loader(batch_size=batch_size)
'''
Train

TODO:
    -Experiment with model architectures and params
    -Decide if we still want to add random noise on top of our random blur
        -Assuming we blur randomly
'''

def train_model(save=False):

    for epoch in range(num_epochs):

        #Loop through batch
        for i, (imgs, tgts) in enumerate(dataloader):

            #Configure Input
            imgs = Variable(imgs.type(Tensor)).to(device=device)
            tgts = Variable(tgts.type(Tensor)).to(device=device)

            #Train fc
            optimizer.zero_grad()

            # Generate a batch of images using fc layer
            gen_imgs = model(imgs)

            # Loss measures generator's ability to fool the discriminator
            loss = loss_func(gen_imgs, tgts)

            loss.backward()
            optimizer.step()

            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
            % (epoch, num_epochs, i, len(dataloader), loss))

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], "../../outputs/out%d.png" % batches_done, nrow=5)
                save_image(tgts.data[:25], "../../outputs/in.png", nrow=5)
                if save: torch.save(model, "model_state.pt")
    
            
            
            
            
            
 
  
            
            
            
            
            
            
        