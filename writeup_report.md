# **Behavioral Cloning** 

## Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imagens_para_writeup/Capturar.JPG "Model Visualization"
[image2]: ./imagens_para_writeup/Centro.jpg "Center Image"
[image3]: ./imagens_para_writeup/Esq.jpg "Recovery Image"
[image4]: ./imagens_para_writeup/Dir.jpg "Recovery Image"
[image5]: ./imagens_para_writeup/DirEsq.jpg "Flipped Image"
[image6]: ./imagens_para_writeup/Centro2.jpg "Second Track"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists in 5 conv layers, followed by 4 fully connected layers (100, 50, 10, 1). It is presented on model.py, lines 39-55. All the steps from the Nvidia's team ["End-to-End Learning for Self-Driving Cars"](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) were followed. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 50 and 53). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 59, "validation_split=0.2"). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

Adam optimizer was used, so the learning rate was not tuned manually (model.py line 58).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Part of the "go forward" data was disconsidered, to equalize the data's histogram. Also, a joystick was used to record every data, in order to have more precision on the steering angle. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia's ["End-to-End Learning for Self-Driving Cars"](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I thought this model might be appropriate because there was a great research on this topic to get to that point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model. I added two Dropout layers.

Then, I reduced epochs and added a Cropping2D layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I augmented the data, using the left and right cameras. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture:

![alt text][image1]

Image from ["End-to-End Learning for Self-Driving Cars"](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

To augment the data sat, I also flipped images and angles to augment the data.

![alt_text][image6]


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to converge to the center. 

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

![alt text][image4]
![alt text][image5]


After the collection process, I had 15.786 number of data points. I then preprocessed this data by Cropping part of the images.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.
