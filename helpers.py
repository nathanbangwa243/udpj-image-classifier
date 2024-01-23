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

def load_category_names(filename):
    """
        Load label mapping

        Parameters:
            filename - Json category_names filename
            
        Returns:
            cat_to_name - category_names dict

    """

    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name


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
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(model.classifier[0].in_features, hidden_units)),
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

def save_checkpoints(model, hidden_units, arch, optimizer, epochs, save_dir):
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

    # if checkpoints dirs does'nt exist, make it.
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)

    checkpoint_file = os.path.join(save_dir, 'checkpoint.pth')

    checkpoint = {'hidden_units': hidden_units, 
                  'arch': arch,
                  'epochs': epochs,
                 'class_to_idx': model.class_to_idx, 
                 'optim_state_dict': optimizer.state_dict(),
                 'model_state_dict': model.state_dict(),
                }

    torch.save(checkpoint, checkpoint_file)


def load_checkpoint(filepath):
    """
        Load checkpoint
        
        Parameters:
            filepath - checkpoint file
            
        Returns:
            model - the trained model
        
    """
    checkpoint = torch.load(filepath)
    
    # build model
    model = build_model(arch=checkpoint['arch'], hidden_units=checkpoint['hidden_units'])
    
    # load model state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # attach class map
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # Open the image
    with Image.open(image) as img:
        # Resize the image
        img = img.resize((256, 256))
        
        # Crop out the center 224x224 portion of the image
        left_margin = (img.width - 224) / 2
        bottom_margin = (img.height - 224) / 2
        right_margin = left_margin + 224
        top_margin = bottom_margin + 224
        img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
        
        # Convert color channels to floats 0-1
        img = np.array(img) / 255.0
        
        # Normalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # Reorder dimensions
        img = img.transpose((2, 0, 1))
        
        # Convert to PyTorch tensor
        img = torch.from_numpy(img).float()
        
    return img