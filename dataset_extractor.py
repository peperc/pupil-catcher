import cv2
import os
import time
import logging

import weka.core.jvm as jvm

from lib.pupil_tracker import get_face_parameters, head_tilt, on_threshold_change
from lib.weka_classifier import create_dataset, add_to_dataset, save_dataset


# Create the threshold window
cv2.namedWindow('Threshold')
cv2.createTrackbar('Value', 'Threshold', 0, 255, on_threshold_change)

# Create the eyes tracker window
cv2.namedWindow('Tracker')

# Create the dataset for training
jvm.logger.setLevel(logging.ERROR)
jvm.start(packages=True)
dataset = create_dataset()

# Starts the webcam
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

key = -1
while(key != 27):
    ret, frame = video.read()
    
    # Get face parameters
    frame, thresh, face_array, left_pupil, right_pupil = get_face_parameters(frame)

    if key not in [-1, 0] and face_array is not None:
        angle = head_tilt(face_array)
        add_to_dataset(dataset, key, left_pupil, right_pupil, face_array[27], angle)
        print(f'Added: Key {key}, L pupil {str(left_pupil)}, R pupil {str(right_pupil)}, Face position {face_array[27]}, Face angle {angle}')
        
    # Show the final frames
    cv2.imshow('Tracker', frame)
    cv2.imshow("Threshold", thresh)
    
    # Get next key
    key = cv2.waitKey(1)

# Save the dataset
now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
save_dataset(dataset, os.path.join(os.path.dirname(__file__), f'datasets/{now}.arff'))

jvm.stop()
video.release()
cv2.destroyAllWindows()
