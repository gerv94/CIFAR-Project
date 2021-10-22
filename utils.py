# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 21:23:32 2021

@author: gerv1
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Function to show an image
def imshow(img, unnormalize_transform, one_channel=False, show=False):
    if one_channel:
        img = img.mean(dim=0)
    img = unnormalize_transform(img)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if show:
        plt.show()
        
def add_pr_curve_tensorboard(writer, classes, class_index, test_probs, test_label, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    

def check_data_classes(dataset):
    classes_items = dict()

    for train_item in dataset:
        label = dataset.classes[train_item[1]]
        if label not in classes_items:
            classes_items[label] = 1
        else:
            classes_items[label] += 1
    
    print('Classes items:', classes_items)
        
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels, classes, unnormalize_transform):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        imshow(images[idx].cpu(), unnormalize_transform)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig