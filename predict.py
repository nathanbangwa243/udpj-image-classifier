#!/usr/bin/env python3
# -*- coding: utf-8 -*-
                                                                             
# PROGRAMMER: @nathanbangwa243
# PURPOSE: TPredict flower name from an image using predict.py
#   
#   Basic usage: python predict.py /path/to/image checkpoint
#   
#   Options: 
#       * Return top-K most likely classes: python predict.py input checkpoint --top_k 3
#       * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
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

# helpers
from helpers import build_model
from helpers import save_checkpoints
from helpers import load_checkpoint
from helpers import process_image
from helpers import load_category_names

def get_input_args():
    """
    Retrieves and parses the 5 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 5 command line arguments. 
    
    If the user fails to provide some or all of the 5 arguments, then the default 
    values are used for the missing arguments. 
    
    Command Line Arguments:
      1. input image as input (positional argument)
      2. checkpoint file as checkpoint (positional argument)
      3. top-K most likely classes as --top_k
      4. mapping of categories to real names as --category_names
      5. Use GPU for training as --gpu with default value False

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
        "input", type=str, help="Path to input image."
    )

    # The model to use
    parser.add_argument(
        "checkpoint", type=str, help="The checkpoint file."
    )

    # Path to checkpoints folder
    parser.add_argument(
        "--top_k", type=int, default=3, help="top-K most likely classes. With 3 as default."
    )
    
    # The learning rate
    parser.add_argument(
        "--category_names", type=str, default="cat_to_name.json", help="The mapping of categories to real names. With cat_to_name.json as default."
    )

    # Use GPU for training
    parser.add_argument(
        "--gpu", action='store_true', help="Use GPU for training. With False as default."
    )
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()


def predict(image_path, model, topk, device):
    ''' 
        Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # move model to device
    model.to(device);
    
    # Implement the code to predict the class from an image file
    image = process_image(image_path)
    
    # add batch and move to device
    image = image.view(1, *image.shape)
    image = image.to(device)
    
    with torch.no_grad():
        model.eval()
        
        # do inference
        log_ps = model(image)

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(topk, dim=1)
        
        top_p, top_class = top_p.cpu()[0].numpy(), top_class.cpu()[0].numpy()

        # Invert the dictionary to get a mapping from index to class
        idx_to_class = {idx: cls for cls, idx in model.class_to_idx.items()}

        classes = [idx_to_class[tc] for tc in top_class]
    
    model.train()
    
    return top_p, classes

def display_prediction(top_p, classes, category_names_file):
    """
        Display classifier results

        Parameters:
            top_p - the top probabilities
            classes - the top classes
            category_names_file - The mapping of categories to real names
            
        Returns:
            None
    """

    cat_to_name = load_category_names(filename=category_names_file)

    top_names = [cat_to_name[cls] for cls in classes]

    for i, (name, prob) in enumerate(zip(top_names, top_p)):
        print("{}. {} : {:.3f}%".format(i+1, name, prob * 100))
    
    


def main():
    # retrieve cmd line argument
    print("[RUN] get_input_args", "=" * 50, '\n')
    in_arg = get_input_args()
    print(in_arg, '\n')

    # Use GPU if it's available
    print("[RUN] torch.device", "=" * 50, '\n')
    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu else "cpu")
    print(device, '\n')

    # load trained model
    print("[RUN] load_checkpoint", "=" * 50, '\n')
    model = load_checkpoint(filepath=in_arg.checkpoint)
    print(model)
    print('\n')

    # make prediction
    print("[RUN] predict", "=" * 50, '\n')
    top_p, classes = predict(image_path=in_arg.input, model=model, topk=in_arg.top_k, device=device)
    print('top_p: ', top_p)
    print('classes: ', classes)
    print('\n')

    # display results
    print("[RUN] display_prediction", "=" * 50, '\n')
    display_prediction(top_p, classes, category_names_file=in_arg.category_names)
    print('\n')

if __name__ == "__main__":
    main()

    # python predict.py "flowers/test/1/image_06743.jpg" checkpoints/checkpoint.pth --gpu