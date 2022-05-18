import cv2
import dlib
import time
import os
import logging

import weka.core.jvm as jvm

from lib.pupil_tracker import face_shape_to_array, get_and_draw_pupils, on_threshold_change, rotate_image, FACE_PREDICTOR, FACES_DETECTOR
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

text = ''
key = -1
while(key != 27):
    ret, frame = video.read()

    # Rotate image as head tilt
    # frame = rotate_image(frame)

    # Get the faces in the frame represented as rectangles
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACES_DETECTOR(gray_frame, 1)

    for face in faces:
        # Get the 68 points of the face 
        face_shape = FACE_PREDICTOR(gray_frame, face)
        face_array = face_shape_to_array(face_shape)

        frame, thresh, left_pupil, right_pupil = get_and_draw_pupils(frame, face_array)

        if key != -1:
            add_to_dataset(dataset, key, left_pupil, right_pupil)
            text += chr(key)
            print(f'Instance added to dataset: {key}, {str(left_pupil)}, {str(right_pupil)}')

    # Show the final frames
    cv2.imshow('Tracker', frame)    
    cv2.imshow("Threshold", thresh)
    
    # Get next key
    key = cv2.waitKey(1)

# Save the dataset and the plaintext file
now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
save_dataset(dataset, os.path.join(os.path.dirname(__file__), f'datasets/{now}.arff'))
with open(os.path.join(os.path.dirname(__file__), f'datasets/{now}.txt'), 'w') as f:
    f.write(text)

jvm.stop()
video.release()
cv2.destroyAllWindows()