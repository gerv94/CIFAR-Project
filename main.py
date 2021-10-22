# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:20:42 2021

@author: gerv1
"""

# Main torch dependencies
import torch
import torchvision
import torchvision.transforms as transforms

# Dependencies to display the images
import matplotlib

# Neural Network dependencies
import torch.nn as nn

# Import models
from model import VGG
from trainner import Trainner

import utils

# Import for tensorboard
from torch.utils.tensorboard import SummaryWriter

# To use inline plotting
matplotlib.use('module://matplotlib_inline.backend_inline')



#:::::::::::::::::::::::::: INITIALS ::::::::::::::::::::::::::

use_gpu = True
load_training_model = False
epochs = 4#2 use 100?
batch_size = 8#4 use 64?
num_workers = 2#2

vgg_type = 'VGG8'
PATH = './cifar_net_' + vgg_type +'_.pth'


if __name__ == '__main__':
    
    
    
    #:::::::::::::::::::::::::: INIT ::::::::::::::::::::::::::
    can_use_gpu = use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if can_use_gpu else "cpu")
    print('Runing on: ', device)
    
    print('::PARAMETERS::')
    print('use_gpu:', use_gpu)
    print('load_training_model:', load_training_model)
    print('epochs:', epochs)
    print('batch_size:', batch_size)
    print('num_workers:', num_workers)
    print('::::::::::::::')
    
    if can_use_gpu:
        nb_cuda = torch.cuda.device_count()
        print("Number of cuda devices:", nb_cuda)
    
    # torch.multiprocessing.freeze_support()
    
    
    
    #:::::::::::::::::::::::::: TRANSFORMS ::::::::::::::::::::::::::
        
    # Load and normalize the CIFAR10 training and test datasets using
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tensor_transform)
    mean = train_dataset.data.mean( axis=(0, 1, 2) ) / 255
    std = train_dataset.data.std( axis=(0, 1, 2) ) / 255
    
    print('Shape:', train_dataset.data.shape)
    print('Mean:', mean)
    print('Standard deviation:', std)
    
    normalize = transforms.Normalize(tuple(mean), tuple(std))
    
    # Define the transforms
    train_transform = transforms.Compose([
        # randomly flips an image with a probability of 50%,
        transforms.RandomHorizontalFlip(), 
        # pads an image by 4 pixel on each side then randomly crops 32x32 from the image after padding.
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        # Those add noise to the data to prevent our model from overfitting
        #transforms.Resize(224), # From paper VGG here trained with 224 * 224 images
        transforms.ToTensor(),
        normalize
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    unnormalize_transform = transforms.Normalize(
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std])
    
    
    
    #:::::::::::::::::::::::::: DATASETS ::::::::::::::::::::::::::
    
    # Retrieve training and test datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # num_workers generates batches in parallel. It essentially prepares the next n batches after a batch has been used
    # pin_memory helps speed up the transfer of data from the CPU to the GPU
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=can_use_gpu)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=can_use_gpu)
    
    
    
    #:::::::::::::::::::::::::: DATASETS ::::::::::::::::::::::::::
    # Get dataset classes
    classes = trainset.classes
    print('Classes:', classes)

    # Check our test and train data.
    print('Checking train set:')
    utils.check_data_classes(trainset)
    print('Checking test set:')
    utils.check_data_classes(testset)
    
    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    # show images and print labels
    utils.imshow(torchvision.utils.make_grid(images), unnormalize_transform, show=True)
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))



    #:::::::::::::::::::::::::: NETWORK ::::::::::::::::::::::::::
    # Initialize NN
    print('Initialize NN')
    #net = Net()
    net = VGG(in_channels=3, num_classes=len(classes), vgg_type=vgg_type)
    
    # Use multiple GPU if available
    if torch.cuda.device_count() > 1:
        print("Found", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net)
    
    # Transfer the NN to the CPU or GPU (not requiered for only CPU)
    print('Transfer the NN to the device')
    net.to(device)
    
    # Load previous state dict if required
    if (load_training_model):
        print('Load previous state ')
        net.load_state_dict(torch.load(PATH))
        print('Loaded previous training model')
        
    # default `log_dir` is "runs" - we'll be more specific here
    print('Init tensorboard writter')
    writer = SummaryWriter('runs/cifar10_experiment_' + vgg_type)
    
    
    
    #:::::::::::::::::::::::::: TRAINING ::::::::::::::::::::::::::
    # Start training 
    trainner = Trainner(net, device, writer)
    trainner.train(epochs, trainloader)
    
    # Store new training model
    torch.save(net.state_dict(), PATH)
    
    trainner.single_test(testloader, classes, batch_size, unnormalize_transform)
    
    
    
    #:::::::::::::::::::::::::: TESTING ::::::::::::::::::::::::::
    # Run testing
    trainner.test(testloader, classes)
        
        
# Agregar mas metricas
# Matriz de confusión
# SK learn (Reporte de clasificación)
# Deep learning book (capitulos: 1 y 8)
