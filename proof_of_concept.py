from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input
import numpy as np

import cv2
import os
import sys

max_size = (750, 750, 3)
center = (max_size[0]/2, max_size[1]/2)

len_args = len(sys.argv)
args = sys.argv

target_path = ''
img_paths = []

if(len_args != 8):
	sys.exit()
else:
	target_path += args[1]
	for i in range(6):
		img_paths.append(args[i+2])

target = cv2.imread(target_path)
target = cv2.resize(target, (224,224), interpolation = cv2.INTER_AREA)
h, w = target.shape[:-1]
augmented = np.zeros(max_size)

# Model to compare feature vectors
resnet_model = ResNet50(weights = 'imagenet', include_top = False)

# Put target on black bg for object detection
if h%2==0 and w%2==0:
	augmented[int(center[0] - h/2):int(center[0] + h/2), int(center[1] - w/2):int(center[1] + w/2), :] = target
elif h%2==1 and w%2==0:
	augmented[int(center[0] - h/2):int(center[0] + h/2), int(center[1] - w/2):int(center[1] + w/2), :] = target
elif h%2==0 and w%2==1:
	augmented[int(center[0] - h/2):int(center[0] + h/2), int(center[1] - w/2):int(center[1] + w/2), :] = target
else:
	augmented[int(center[0] - h/2):int(center[0] + h/2), int(center[1] - w/2):int(center[1] + w/2), :] = target

cv2.imwrite('data/augmented.png', augmented)

# Run YOLO to obtain tag of the target
f = open('coords.txt', 'w+')
f.close()
os.system('./darknet detect cfg/yolov3.cfg yolov3.weights data/augmented.png')

# Find the target with largest area (of bounding box)
coords_file = open('coords.txt', 'r')
candidate_targets = []
for lines in coords_file:
	fields = lines.split(',')
	tag = fields[0]
	confidence = float(fields[1])
	left = int(fields[2])
	top = int(fields[3])
	right = int(fields[4])
	bottom = int(fields[5])
	candidate_targets.append([tag, (bottom-top)*(right-left)])

target_tag = candidate_targets[0][0]
max_area = candidate_targets[0][1]

for candidate_target in candidate_targets:
	if candidate_target[1] > max_area:
		target_tag = candidate_target[0]
		max_area = candidate_target[1]

print('\ntarget_tag is successfully stored!')
print('target_tag is ' + target_tag +'!')
# target_tag is successfully stored 

# Find and store feature vector of target
target = np.expand_dims(target, axis = 0)
target_features = resnet_model.predict(target)

# Run following steps for each frame
frames_with_target = []
for i in range(6):
	# Retrieve image to search target in
	image_to_detect = cv2.imread(img_paths[i])

	# Run YOLO to obtain regions of interest in image
	os.remove('coords.txt')
	f = open('coords.txt', 'w+')
	f.close()
	os.system('./darknet detect cfg/yolov3.cfg yolov3.weights ' + img_paths[i])

	# Use the coords of ROIs detected
	detection_file = open('coords.txt', 'r')
	rois = []
	for lines in detection_file:
		fields = lines.split(',')
		tag = fields[0]
		confidence = float(fields[1])
		left = int(fields[2])
		top = int(fields[3])
		right = int(fields[4])
		bottom = int(fields[5])
		if tag == target_tag:
			roi = np.copy(image_to_detect[top:bottom, left:right, :])
			#cv2.imshow('roi', roi)
			#cv2.waitKey(0)
			rois.append([roi, left, top, right, bottom])

	print('Candidate ROIs retrieved...')
	print('No. of ROIs: ' + str(len(rois)))
	# Retrieved candidate rois

	# Extract feature vectors of rois and find best match using Mean Squared Error as a prototype error norm
	if len(rois) > 0:
		min_err = 0
		best_roi = None
		iteration = 0
		for roi in rois:
			roi_resized = cv2.resize(roi[0], (224,224), interpolation = cv2.INTER_AREA)
			roi_resized = np.expand_dims(roi_resized, axis = 0)
			roi_features = resnet_model.predict(roi_resized) 
			err = np.sum((target_features - roi_features)**2)
			if iteration == 0:
				min_err = err
				best_roi = roi
			else:
				if err < min_err:
					min_err = err
					best_roi = roi

		print('Best match of frame ', str(i), ' found at: ', (left, top))
		frames_with_target.append([best_roi, min_err, i])
		#cv2.imshow('best match', best_roi)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
	else:
		print('No good match found in frame ', str(i), '!')

# Find best of all matches in each frame
best_frame_with_roi = None
iteration = 0
if len(frames_with_target) > 0:
	for frame in frames_with_target:
		if iteration == 0:
			best_frame_with_roi = frame
		else:
			if best_frame_with_roi[1] > frame[1]:
				best_frame_with_roi = frame
	print('Target found in frame ', best_frame_with_roi[2], 'successfully!!')
	frame_detected = cv2.imread(img_paths[best_frame_with_roi[2]])
	cv2.rectangle(frame_detected, (best_frame_with_roi[0][1], best_frame_with_roi[0][2]), (best_frame_with_roi[0][3], best_frame_with_roi[0][4]), (255,0,0), 2)
	cv2.imshow('best match', frame_detected)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
else:
	print('Target not found anywhere!!!')




