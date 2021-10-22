# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:48:49 2021

@author: gerv1
"""
# Main torch dependency
import torch
# Neural Network Model dependencies
import torch.nn as nn
# Dependencies for the optimizer
import torch.optim as optim

import torch.nn.functional as F

import torchvision

# To monitor time
import time

import utils

class Trainner():
    def __init__(self, net, device, tensorboard_writer=None):
        self.net = net
        self.device = device
        self.writer = tensorboard_writer
    
    def train(self, epochs, trainloader, lr=0.001, momentum=0.9):
        
        # From paper: Softmax is included in nn.CrossEntropyLoss
        loss_function = nn.CrossEntropyLoss() # Loss function
        optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum) # Implements stochastic gradient descent 
        
        # Start Training
        print('Started training')
        start_time = time.time()
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0
            correct = 0
            total = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
        
                # print statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    accuracy = 100.0 * correct / total
                    
                    print('[%d, %5d] training loss: %.3f accuracy: %.3f' % (epoch + 1, i + 1, running_loss / 2000, accuracy))
                    
                    # ...log the running loss to tensorboard
                    if (self.writer != None):
                        self.writer.add_scalar('training loss', running_loss / 2000, epoch * len(trainloader) + i)
                        self.writer.add_scalar('training accuracy', accuracy, epoch * len(trainloader) + i)
                    
                    # ...log a Matplotlib Figure showing the model's predictions on a random mini-batch
                    #self.writer.add_figure('predictions vs. actuals', plot_classes_preds(net, inputs, labels, classes, unnormalize_transform), global_step=epoch * len(trainloader) + i)
                    running_loss = 0.0
        print('Time: ', (time.time() - start_time), 'seconds')
        print('Finished Training')
    
    @torch.no_grad()
    def test(self, testloader, classes):
        correct = 0
        total = 0
        class_probs = []
        class_label = []
        
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        
        for data in testloader:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            # calculate outputs by running images through the network
            outputs = self.net(images)
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
            class_probs.append(class_probs_batch)
            class_label.append(labels)
            
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

        
        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                           accuracy))
        
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        
        if self.writer != None:
            test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
            test_label = torch.cat(class_label)
        
            # plot all the pr curves
            for i in range(len(classes)):
                utils.add_pr_curve_tensorboard(self.writer, classes, i, test_probs, test_label, global_step=0)
                
    
    
    def single_test(self, testloader, classes, batch_size, unnormalize_transform):
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        
        # print images
        utils.imshow(torchvision.utils.make_grid(images), unnormalize_transform, show=True)
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
        
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self.net(images)
        
        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                      for j in range(batch_size)))