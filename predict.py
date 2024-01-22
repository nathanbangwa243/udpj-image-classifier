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

def main():
    # retrieve cmd line argument
    print("[RUN] get_input_args", "=" * 50, '\n')
    in_arg = get_input_args()
    print(in_arg, '\n')

if __name__ == "__main__":
    main()