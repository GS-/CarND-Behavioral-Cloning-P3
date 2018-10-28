# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Video References)

[video1]: ./video.mp4 "Result at 30mph"
[video2]: ./video1.mp4 "Result at 20mph"

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```


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
