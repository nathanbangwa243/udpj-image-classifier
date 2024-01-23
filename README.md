# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

## My Result

![Loss Metrics](assets/train_valid.png)

PROGRAMMER: @nathanbangwa243

## 0. download_dataset.sh

PURPOSE: Download flowers dataset

* **Make the script executable using the following command:**
 
```bash
chmod +x download_dataset.sh

```

* **Now, you can run the script:**
 
```bash
./download_dataset.sh

```

## 1. train.py

PURPOSE: Train a new network on a data set using train.py

### Basic usage: 

```python
python train.py data_directory
```

Prints out training loss, validation loss, and validation accuracy as the network trains

### Options:

* **Set directory to save checkpoints:** 

```python
python train.py data_dir --save_dir save_directory
```

* **Choose architecture:** 

```python
python train.py data_dir --arch "vgg13"
```

* **Set hyperparameters:** 

```python
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 10
```

* **Use GPU for training:**

```python
python train.py data_dir --gpu
```

### Concret Example

* **Use model trained with notebook**
```python 
python predict.py "flowers/test/1/image_06743.jpg" checkpoints/checkpoint-final.pth --gpu
```

## 2. predict.py

PURPOSE: TPredict flower name from an image using predict.py
   
### Basic usage: 

```python
python predict.py /path/to/image checkpoint
```

### Options: 

* **Return top-K most likely classes:** 

```python
python predict.py input checkpoint --top_k 3
```

* **Use a mapping of categories to real names:** 
 
```python
python predict.py input checkpoint --category_names cat_to_name.json
```

* **Use GPU for inference:** 
 
```python
python predict.py input checkpoint --gpu
```

### Concret Example

* **Use model trained with notebook**
```python 
python predict.py "flowers/test/1/image_06743.jpg" checkpoints/checkpoint-final.pth --gpu
```

* **Use model trained with train.py**
```python 
python predict.py "flowers/test/1/image_06743.jpg" checkpoints/checkpoint.pth --gpu
```