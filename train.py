#!/usr/bin/env python3
# -*- coding: utf-8 -*-
                                                                             
# PROGRAMMER: @nathanbangwa243
# PURPOSE: Train a new network on a data set using train.py
#   
#   Basic usage: python train.py data_directory
#       Prints out training loss, validation loss, and validation accuracy as the network trains
#   
#   Options: 
#       * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory 
#       * Choose architecture: python train.py data_dir --arch "vgg13" 
#       * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#       * Use GPU for training: python train.py data_dir --gpu

# Imports python modules
import argparse

import os
import json
import numpy as np
from collections import OrderedDict
from datetime import datetime

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def get_input_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 7 command line arguments. 
    
    If the user fails to provide some or all of the 7 arguments, then the default 
    values are used for the missing arguments. 
    
    Command Line Arguments:
      1. Image Folder as data_dir (positional argument)
      2. Checkpoints folder as --save_dir with default value 'checkpoints'
      3. Model Architecture as --arch with default value 'vgg13'
      4. Learning rate as --learning_rate with default value 0.003
      5. Number of hidden units as --hidden_units with default value 512
      6. Number of epochs as --epochs with default value 5
      7. Use GPU for training as --gpu with default value False

    This function returns these arguments as an ArgumentParser object.
    
    Parameters:
     None - simply using argparse module to create & store command line arguments
    
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 4 command line arguments as mentioned above using add_argument() from ArguementParser method

    # Path to images folder
    parser.add_argument(
        "data_dir", type=str, help="Path to images folder."
    )

    # Path to checkpoints folder
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints/", help="Path to checkpoints folder. With 'checkpoints/' as default."
    )

    # The model to use
    parser.add_argument(
        "--arch", type=str, default="vgg13", choices=["vgg13", "resnet50"], help="The model to use. With 'vgg13' as default."
    )

    # The learning rate
    parser.add_argument(
        "--learning_rate", type=float, default=0.003, help="The learning rate. With 0.003 as default."
    )

    # Number of hidden units
    parser.add_argument(
        "--hidden_units", type=int, default=512, help="Number of hidden units. With 512 as default."
    )

    # Number of hidden units
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs. With 5 as default."
    )

    # Use GPU for training
    parser.add_argument(
        "--gpu", action='store_true', help="Use GPU for training. With False as default."
    )
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()


def get_dataloaders(data_dir):
    """
        Define your transforms for the training, validation, and testing sets

        Parameters:
            data_dir - Path to images folder
            
        Returns:
            dataloaders - the dataloarders dict
    """

    # subdirs
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {"train": transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),
                    "test": transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
                    }

    data_transforms["valid"] = data_transforms["test"]


    # Load the datasets with ImageFolder
    image_datasets = {"train": datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                    "test": datasets.ImageFolder(test_dir, transform=data_transforms['test']),
                    "valid": datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {"train": torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
                "test": torch.utils.data.DataLoader(image_datasets['test'], batch_size=32),
                "valid": torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
                }
    
    return dataloaders

def build_model(arch, hidden_units, nb_classes=102):
    """
        Download the pretrained model and attach new classifier.

        Parameters:
            arch - pretrained model to download
            hidden_units - number of hidden unit of the new classifier
            nb_classes - output of the new classifier
            
        Returns:
            torchvision.models.vgg.VGG - the model
    """

    model = None

    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        # define new classifier 
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(model.classifier.in_features, hidden_units)),
                                                ('relu', nn.ReLU()),
                                                ('dropout', nn.Dropout(0.2)),
                                                ('fc2', nn.Linear(hidden_units, nb_classes)),
                                                ('output', nn.LogSoftmax(dim=1))
                                            ]))
        
    
        # attach new classifier
        model.classifier = classifier
    
    else:
        model = models.resnet50(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        # define new classifier 
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(model.fc.in_features, hidden_units)),
                                                ('relu', nn.ReLU()),
                                                ('dropout', nn.Dropout(0.2)),
                                                ('fc2', nn.Linear(hidden_units, nb_classes)),
                                                ('output', nn.LogSoftmax(dim=1))
                                            ]))
        
    
        # attach new classifier
        model.fc = classifier
    
    return model

def train_model(model, dataloaders, learnrate, epochs, device):
    """
        Train model.

        Parameters:
            model - then model to train
            dataloaders - the dataloarders dict
            learnrate - the learning rate
            device - torch.device
            
        Returns:
            criterion - the loss function
            optimizer - the optimizer
    """

    # define loss function
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.fc.parameters(), lr=learnrate)

    # move model to device
    model.to(device);

    train_losses, valid_losses = [], []

    for e in range(epochs):
        tot_train_loss = 0
        for images, labels in dataloaders['train']:
            optimizer.zero_grad()

            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            tot_train_loss += loss.item()

            loss.backward()
            optimizer.step()
        else:
            tot_valid_loss = 0
            valid_correct = 0  # Number of correct predictions on the valid set

            # switch model in evaluation mode
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for images, labels in dataloaders['valid']:
                    # Move input and label tensors to the default device
                    images, labels = images.to(device), labels.to(device)

                    log_ps = model(images)
                    loss = criterion(log_ps, labels)
                    tot_valid_loss += loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    valid_correct += equals.sum().item()

            # switch model in training mode
            model.train()

            # Get mean loss to enable comparison between train and valid sets
            train_loss = tot_train_loss / len(dataloaders['train'].dataset)
            valid_loss = tot_valid_loss / len(dataloaders['valid'].dataset)

            # At completion of epoch
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Validation Loss: {:.3f}.. ".format(valid_loss),
                  "Validation Accuracy: {:.3f}".format(valid_correct / len(dataloaders['valid'].dataset)))
    
    return criterion, optimizer


def test_model(model, dataloaders, criterion, device):
    """
        Download the pretrained model and attach new classifier.

        Parameters:
            model - the trained model
            dataloaders - the dataloarders dict
            criterion - the loss function
            device - torch.device
            
        Returns:
            criterion - the loss function
            optimizer - the optimizer
    """
    test_losses = []

    tot_test_loss = 0
    test_correct = 0  # Number of correct predictions on the test set

    # switch model in evaluation mode
    model.eval()

    # Turn off gradients for testation, saves memory and computations
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            tot_test_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_correct += equals.sum().item()

    # switch model in training mode
    # model.train()

    # Get mean loss to enable comparison between train and test sets
    test_loss = tot_test_loss / len(dataloaders['test'].dataset)

    # At completion of epoch
    test_losses.append(test_loss)

    print("Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(train_loss),
            "Test Loss: {:.3f}.. ".format(test_loss),
            "Test Accuracy: {:.3f}".format(test_correct / len(dataloaders['test'].dataset)))



def save_checkpoints(model, optimizer, epochs, save_dir):
    """
        Save checkpoints

        Parameters:
            model - the trained model
            optimizer - the optimizer
            epochs - the number of epochs
            save_dir - Checkpoints folder
            
        Returns:
            None

    """
    checkpoint_file = save_dir + 'checkpoint.pth'

    checkpoint = {'epochs': epochs,
                 'class_to_idx': model.class_to_idx, 
                 'optim_state_dict': optimizer.state_dict(),
                 'model_state_dict': model.state_dict(),
                }

    torch.save(checkpoint, checkpoint_file)

def main():
    # retrieve cmd line argument
    print("[RUN] get_input_args", "=" * 50, '\n')
    in_arg = get_input_args()
    print(in_arg, '\n')

    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu else "cpu")

    # get dataloaders
    dataloaders = get_dataloaders(data_dir=in_arg.data_dir)

    # build_model
    print("[RUN] build_model", "=" * 50, '\n')
    model = build_model(arch=in_arg.arch, hidden_units=in_arg.hidden_units)
    print(model, '\n')


    # train model
    print("[RUN] train_model", "=" * 50, '\n')
    criterion, optimizer = train_model(model, dataloaders, learnrate=in_arg.learning_rate, epochs=in_arg.epochs)
    print(criterion, optimizer, '\n')
    
    # test model
    print("[RUN] test_model", "=" * 50, '\n')
    test_model(model, dataloaders, criterion, device)
    print('\n')

    # save model
    print("[RUN] test_model", "=" * 50, '\n')
    save_checkpoints(model, optimizer, in_arg.epochs, in_arg.save_dir)
    print('\n')


if __name__ == "__main__":
    main()
