import cv2
import dlib
import time
import os

import weka.core.jvm as jvm

from weka.core.converters import Saver
from lib.pupil_tracker import face_shape_to_array, get_and_draw_pupils, on_threshold_change
from lib.weka_classifier import create_dataset, add_to_dataset


# With the detector, we get faces from frames represented as rectangles 
faces_detector = dlib.get_frontal_face_detector()
# With the predictor, we get the 68 points representing the face from the face_detector's rectangles
face_predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'lib/models/shape_68.dat'))

# Create the threshold window
cv2.namedWindow('Threshold')
cv2.createTrackbar('Value', 'Threshold', 0, 255, on_threshold_change)

# Create the eyes tracker window
cv2.namedWindow('Tracker')

# Create the dataset for training
jvm.start()
dataset = create_dataset()

# Starts the webcam
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

key = -1
while(key != 27):
    ret, frame = video.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get the faces in the frame represented as rectangles
    faces = faces_detector(gray_frame, 1)
    for face in faces:
        # Get the 68 points of the face 
        face_shape = face_predictor(gray_frame, face)
        face_array = face_shape_to_array(face_shape)

        frame, thresh, left_eye, right_eye = get_and_draw_pupils(frame, face_array)

        if key != -1:
            add_to_dataset(dataset, key, left_eye, right_eye)
            print(f'Instance added to dataset: {key}, {str(left_eye)}, {str(right_eye)}')

    # Show the final frames
    cv2.imshow('Tracker', frame)
    cv2.imshow("Threshold", thresh)
    
    key = cv2.waitKey(1)

t = time.localtime()
current_time = time.strftime("%H-%M-%S", t)

saver = Saver(classname="weka.core.converters.ArffSaver")
saver.save_file(dataset, os.path.join(os.path.dirname(__file__), f'datasets/{current_time}.arff'))

jvm.stop()
video.release()
cv2.destroyAllWindows()