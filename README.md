# BehavioralCloning
A project to train a neural network to drive in a car simulator

## Overview
This project is contained in three main files: preprocess.py, model.py, and drive.py. The preprocess.py augments the entire dataset and stores it in a numpy array (.npy) file, which was not uploaded due to large size. Model.py reads in the dataset and trains the model, then stores the model in model.h5. Drive.py then drives the car in the simulator based on the predicted steering angle from the model.

## Process
# The Data
The data consists of one and a half laps. I originally recorded almost 15 gigabytes of data, however the car still failed to drive around steep turns. I decided to decrease the amount of data so it fit into my RAM better. I chose to record fewer laps at a higher quality than many sloppy laps. I recorded one complete lap and one lap consisting solely of sharp turns. This did not decrease the accuracy of the model significantly from the previous large dataset.

# Preprocess
I moved the preprocessing steps to a separate file because it took a long time and it only needed to be run once. This sped up the process of training the model as the dataset did not need to be processed every time I changed a hyperparameter. The preprocessing augments the dataset and creates six times as many images as the original dataset. It also resizes the images so that they will be square after they are cropped in the model. The script uses the CSV file to compile a list of images from all three cameras and a list of measurements with a correction factor for the left and right. It also adds a flipped version of every image with the appropriate flipped measurement to prevent a bias towards steering left or right. These images are converted to a Numpy Float32 array and stored in a .npy file for use with the model.

# The Model
I chose to use the NVIDIA model, as my research showed that it was a simple model that worked well with behavioral cloning. It consists of two preprocessing layers that normalize the images to between -1 and 1 by dividing by 127.5 and subtracting 1. It then crops the top and bottom of the image so that only the relevant parts of the image are fed to the model. The model itself consists of the following architecture (Created from model.summary() function):
_________________________________________________________________
Model: "sequential"
| Layer (type)            | Output Shape        | Param # |
|-------------------------|---------------------|---------|
| cropping2d (Cropping2D) | (None, 100, 100, 3) | 0       |
| lambda (Lambda)         | (None, 100, 100, 3) | 0       |
| conv2d (Conv2D)         | (None, 48, 48, 24)  | 1824    |
| conv2d_1 (Conv2D)       | (None, 22, 22, 36)  | 21636   |
| conv2d_2 (Conv2D)       | (None, 9, 9, 48)    | 43248   |
| conv2d_3 (Conv2D)       | (None, 7, 7, 64)    | 27712   |
| conv2d_4 (Conv2D)       | (None, 5, 5, 64)    | 36928   |
| flatten (Flatten)       | (None, 1600)        | 0       |
| dense (Dense)           | (None, 100)         | 160100  |
| dropout (Dropout)       | (None, 100)         | 0       |
| dense_1 (Dense)         | (None, 50)          | 5050    |
| dropout_1 (Dropout)     | (None, 50)          | 0       |
| dense_2 (Dense)         | (None, 10)          | 510     |
| dropout_2 (Dropout)     | (None, 10)          | 0       |
| dense_3 (Dense)         | (None, 1)           | 11      |
Total params: 297,019
Trainable params: 297,019
Non-trainable params: 0
_________________________________________________________________
A dropout of 0.5 is used to prevent overfitting. The model has a loss of about 0.03 on the training set and 0.04 on the validation set. The model is compiled with an Adam compiler with a starting rate of 0.001. Loss is calculated with the mean squared error. The model uses 20% of the data for validation and runs for 10 epochs with a batch size of 1500, which is the most I could use with my GPU. An early stopping callback is used on the validation loss with a patience of 2.

# Driving
The default driving script is used, with a one minor change. It uses the same openCV function as the preprocessing script to resize the images to the appropriate size for the model.

