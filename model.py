# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:21:27 2021

@author: gerv1
"""

# Neural Network Model dependencies
import torch.nn as nn

# New model - from VGG paper (VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION)
# Visual Geometry Group, Department of Engineering Science, University of Oxford
VGG_types = {
    'VGG8' : [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1000, vgg_type='VGG11'):
        super(VGG, self).__init__() # run the init of the parent method
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types[vgg_type])
        
        # From paper:  followed by three Fully-Connected (FC) layers: the first two have 4096 channels each, the third performs 1000-way ILSVRC classification and thus contains 1000 channels (one for each class) 
        self.fcs = nn.Sequential(
            #nn.Linear(512 * 7 * 7, 4096), # (224 / ( 2 ** num_max_pool_layers ))
            nn.Linear(512, 4096), # because we are using crops of 32 x 32
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, num_classes))
        # Dropout is a simple technique that will randomly drop nodes out of the network. 
        # It has a regularizing effect as the remaining nodes must adapt to pick-up the slack of the removed nodes.

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels 
        
        for i in architecture:
            if type(i) == int:
                out_channels = i
                # From paper: The convolution stride is fixed to 1 pixel; the spatial padding of conv. layer input is such that the spatial resolution is preserved after convolution, i.e. the padding is 1 pixel for 3 × 3 conv. layers.
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(i), # Not included on original vgg paper
                           nn.ReLU()] # From paper: All hidden layers are equipped with the rectification (ReLU)
                # Update the in_channels for the next convolution layer
                in_channels = i
            elif i == 'M':
                # From paper: Max-pooling is performed over a 2 × 2 pixel window, with stride 2
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)