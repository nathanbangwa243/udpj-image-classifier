#!/bin/bash

wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
unlink flowers
mkdir flowers && tar -xzf flower_data.tar.gz -C flowers