# DC GAN CelebA
Keras DC GAN CelebA

## Overview
DC GAN that is written using [Keras library](https://keras.io/) and trained on [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). It generates human faces.
Though the code was drastically overwriten, the baseline was taken from [another repository](https://github.com/YongWookHa/DCGAN-Keras)

## Required project folder structure
The following folders should be present in the project:
- src/datasets/CelebA (folder where CelebA dataset should reside)
- src/datasets/CelebA/<img_size>x<img_size> (folder with resized CelebA images)
- src/models (folder that will contain saved trained models)
- samples (folder with sampled images generated by the GAN during training)

## Usage
Install all third-party modules from requirements.txt file

You might need to resize the CelebA images. During my tests I was using 64x64 image size.
To do so you need to run crop_face() function.

You can change the main parameters in lines 23-29 if you like.
Changing the DEBUG constant to 0 will switch the model into debug mode so it will train only 2 epochs.
NUM_OF_EPOCHS constant sets the number of epochs for the GAN training.
Run the code to get some generated faces.

## Examples
- result after 30 epochs:
![Example](https://github.com/cr0mwell/GAN-CelebA/blob/master/example.png?raw=true)