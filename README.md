# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files:
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :).

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Here is a detailed look at my model:

```
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 70, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 34, 159, 24)       672
_________________________________________________________________
dropout_1 (Dropout)          (None, 34, 159, 24)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 31, 156, 48)       18480
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 14, 76, 96)        115296
_________________________________________________________________
dropout_2 (Dropout)          (None, 14, 76, 96)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 38, 96)         0
_________________________________________________________________
dropout_3 (Dropout)          (None, 7, 38, 96)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 36, 128)        110720
_________________________________________________________________
dropout_4 (Dropout)          (None, 5, 36, 128)        0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 33, 256)        524544
_________________________________________________________________
dropout_5 (Dropout)          (None, 2, 33, 256)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 1, 16, 256)        0
_________________________________________________________________
dropout_6 (Dropout)          (None, 1, 16, 256)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 4096)              0
_________________________________________________________________
dense_1 (Dense)              (None, 2000)              8194000
_________________________________________________________________
activation_1 (Activation)    (None, 2000)              0
_________________________________________________________________
dense_2 (Dense)              (None, 1000)              2001000
_________________________________________________________________
activation_2 (Activation)    (None, 1000)              0
_________________________________________________________________
dropout_7 (Dropout)          (None, 1000)              0
_________________________________________________________________
dense_3 (Dense)              (None, 500)               500500
_________________________________________________________________
dense_4 (Dense)              (None, 100)               50100
_________________________________________________________________
activation_3 (Activation)    (None, 100)               0
_________________________________________________________________
dense_5 (Dense)              (None, 50)                5050
_________________________________________________________________
activation_4 (Activation)    (None, 50)                0
_________________________________________________________________
dropout_8 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_6 (Dense)              (None, 10)                510
_________________________________________________________________
activation_5 (Activation)    (None, 10)                0
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 11
=================================================================
Total params: 11,520,883
Trainable params: 11,520,883
Non-trainable params: 0
_________________________________________________________________
```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. See above for details.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 175).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also drove the car clockwise to balance out the left turn biases. During the clockwise run, I also collected data recovering from the left and right sides of the road.


#### 5. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to move back to the center of the lane.

I also corrected data by driving the opposite direction on the same track. This was done for collect data on two laps. I also colleced recovery data while driving in the opposite direction.

To train the model I used a set of 18K images. During training I also performed following steps to decrease the loss for the model.

* Each image was normalized to remove noise.
* Each image was flipped and added back to the training set with reversed angle to augment the training set. This essentially double the number of images available to train from 18K+ to 36K+.
* I also cropped the image so that we training can focus on the lanes and ignore extra noise.
* I didn't have to incorporate the left and right image into my model. My model have a fairly high accuracy with left and right images.

### Result Run

Video of car driving in autonomous mode:
![alt text][video1]


