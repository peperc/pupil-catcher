import cv2
import dlib
import os
import time
import logging

import weka.core.jvm as jvm

from lib.pupil_tracker import face_shape_to_array, get_and_draw_pupils, threshold_finder
from lib.weka_classifier import create_dataset, add_to_dataset, save_dataset


# With the detector, we get faces from frames represented as rectangles 
faces_detector = dlib.get_frontal_face_detector()
# With the predictor, we get the 68 points representing the face from the face_detector's rectangles
face_predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'lib/models/shape_68.dat'))

# Gets the video and it's frames count
video = cv2.VideoCapture('shots/3.mp4')
total_frames= int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1
ret, frame = video.read()

# Create the dataset for later classification
jvm.logger.setLevel(logging.ERROR)
jvm.start(packages=True)
dataset = create_dataset()

# Find a optimal threshold
threshold_finder(frame)

i = 1
ret = True
while(ret):
    ret, frame = video.read()
    if ret:
        # Get the faces in the frame represented as rectangles
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faces_detector(gray_frame, 1)
        
        for face in faces:
            # Get the 68 points of the face 
            face_shape = face_predictor(gray_frame, face)
            face_array = face_shape_to_array(face_shape)

            frame, thres, left_pupil, right_pupil = get_and_draw_pupils(frame, face_array)
            add_to_dataset(dataset, None, left_pupil, right_pupil)

        print(f'Processed frame {i} out of {total_frames}')
        i += 1

# Save the dataset
save_dataset(dataset, os.path.join(os.path.dirname(__file__), f'datasets/{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.arff'))

video.release()
