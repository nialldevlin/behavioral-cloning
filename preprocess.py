import csv
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

print("starting")

#Open CSV file
lines = []
with open("data/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

#Create the dataset
images = []
steering_angles = []
for row in lines:
	steering_center = float(row[3])

	#Calculate correction angles for left and right images
	correction = 0.3
	steering_left = steering_center + correction
	steering_right = steering_center - correction

	#Read images and resize them
	img_center = cv2.imread(row[0])
	img_center = cv2.resize(img_center, dsize=(100, 160))
	img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
	img_left = cv2.imread(row[1])
	img_left = cv2.resize(img_left, dsize=(100, 160))
	img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
	img_right = cv2.imread(row[2])
	img_right = cv2.resize(img_right, dsize=(100, 160))
	img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

	#Augment dataset by flipping all images and steering measurements
	aug_img_center = cv2.flip(img_center, 1)
	aug_steering_center = steering_center * -1.0
	aug_img_left = cv2.flip(img_left, 1)
	aug_steering_left = steering_left * -1.0
	aug_img_right = cv2.flip(img_right, 1)
	aug_steering_right = steering_right * -1.0

	#Append all six images to dataset
	images.extend((img_center, img_left, img_right, aug_img_center, aug_img_left, aug_img_right))
	steering_angles.extend((steering_center, steering_left, steering_right, aug_steering_center, aug_steering_left, aug_steering_right))

#Convert to Numpy array as float32
x_train = np.array(images, dtype=np.float32)
y_train = np.array(steering_angles, dtype=np.float32)

#Save to file
with open('x_train.npy', 'wb') as f:
	np.save(f, x_train)
with open('y_train.npy', 'wb') as f:
	np.save(f, y_train)

print("finished preprocessing")