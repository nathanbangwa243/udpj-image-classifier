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

try:
    from PIL import Image
    import matplotlib.pyplot as plt

    import torch
    from torch import nn
    from torch import optim
    import torch.nn.functional as F
    from torchvision import datasets, transforms, models

except Exception: # when working at 127.0.0.1
    pass


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

def get_number_of_classes(data_dir):
    """
        Compute number of classes base on data_dir.

        Parameters:
            data_dir - Image Folder
            
        Returns:
            Int - the number of classes
    """
    return len(os.listdir(data_dir))

def build_model(arch, hidden_units, nb_classes):
    """
        Download the pretrained model and attach new classifier.

        Parameters:
            arch - pretrained model to download
            hidden_units - number of hidden unit of the new classifier
            nb_classes - output of the new classifier
            
        Returns:
            torch. - the number of classes
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

def main():
    # retrieve cmd line argument
    print("[RUN] get_input_args", "=" * 50, '\n')
    in_arg = get_input_args()
    print(in_arg, '\n')

    # Compute number of classes
    print("[RUN] get_number_of_classes", "=" * 50, '\n')
    nb_classes = get_number_of_classes(data_dir=in_arg.data_dir)
    print("nb_classes: ", nb_classes)

    # build_model
    print("[RUN] get_number_of_classes", "=" * 50, '\n')
    model = build_model(arch=in_arg.arch, hidden_units=in_arg.hidden_units, nb_classes=nb_classes)
    print(model)



if __name__ == "__main__":
    main()
