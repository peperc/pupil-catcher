import cv2
import time
import os
import logging

import numpy as np
import weka.core.jvm as jvm

from lib.pupil_tracker import face_shape_to_array, get_and_draw_pupils, on_threshold_change, rotate_image, head_tilt, FACE_PREDICTOR, FACES_DETECTOR
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
    thresh = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Rotate image as head tilt
    # frame = rotate_image(frame)

    # Get the faces in the frame represented as rectangles
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACES_DETECTOR(gray_frame, 1)

    if faces:
        # Get the 68 points of the face 
        face_shape = FACE_PREDICTOR(gray_frame, faces[0])
        face_array = face_shape_to_array(face_shape)

        frame, thresh, left_pupil, right_pupil = get_and_draw_pupils(frame, face_array)

        if key not in [-1, 0]:
            add_to_dataset(dataset, key, left_pupil, right_pupil, face_array[27], head_tilt(face_array))
            print(f'Instance added to dataset: {key}, {str(left_pupil)}, {str(right_pupil)}')
        

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