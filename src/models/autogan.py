import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AutoEncoder(nn.Module):
    """
    takes in a square image and tries to reconstruct it from a downsampled feature representation
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.mse_loss = nn.MSELoss()
        self.grad_loss = nn.MSELoss()
    
    def forward(self, input_image, mode='train'):
        encoding = self.encoder(input_image)
        
        if mode == 'encode':
            return encoding
        else:
            decoding = self.decoder(encoding)
            if mode == 'decode':
                return decoding
            
            elif mode == 'train':
                input_clone = input_image.clone()
                input_clone.requires_grad = True
                input_mean = torch.mean(input_clone)
                input_mean.backward()
                input_grad = input_clone.grad.data
                input_clone.grad = None

                decoding_clone = decoding.clone()
                decoding_clone.requires_grad = True
                decoding_mean = torch.mean(decoding_clone)
                decoding_mean.backward()
                decoding_grad = decoding_clone.grad.data
                decoding_clone.grad = None

                mse_loss = self.mse_loss(input_image, decoding)
                grad_loss = self.grad_loss(input_grad, decoding_grad)

                loss = mse_loss + grad_loss
                return loss
            else:
                return None
    
    
class Encoder(nn.Module):
    """
    Sub module that takes in an input image and encodes it into a lower dimension vector
    subspace 
    """
    def __init__(self):
        super(Encoder, self).__init__()
        
    
    def forward(self, input_image):
        pass
    
class Decoder(nn.Module):
    """
    submodule that takes a low dimensional feature representation and decodes it back to the original
    image
    """
    def __init__(self):
        super(Decoder, self).__init__()
        
    def forward(self, features):
        pass

class AEGenerator(nn.Module):
    """
    module that generates a blur-invariant dense encoding of a blurred image
    """
    def __init__(self):
        super(AEGenerator, self).__init__()
    
    def forward(self, blurred_image):
        pass
    
class AEDiscriminator(nn.Module):
    """
    discriminator that tries to tell whether or not a low dimension feature representation 
    is from a blur image (generator) or from a sharp image (autoencoder encoding)
    """
    
    def __init__(self):
        super(AEDiscriminator, self).__init__()
    
    def forward(self, x):
        pass


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, out_features, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)