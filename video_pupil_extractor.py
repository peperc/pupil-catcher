import cv2
import dlib
import os
import time

import weka.core.jvm as jvm

from lib.pupil_tracker import face_shape_to_array, get_and_draw_pupils, on_threshold_change
from lib.weka_classifier import create_dataset, add_to_dataset, save_dataset


# With the detector, we get faces from frames represented as rectangles 
faces_detector = dlib.get_frontal_face_detector()
# With the predictor, we get the 68 points representing the face from the face_detector's rectangles
face_predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'lib/models/shape_68.dat'))

# Gets the video and it's frames count
video = cv2.VideoCapture('shots/2.mp4')
total_frames= int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Create the dataset for later classification
jvm.start()
dataset = create_dataset()

# Find a way to change this
on_threshold_change(79)

i = 1
while(video.isOpened()):
    ret, frame = video.read()
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Get the faces in the frame represented as rectangles
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
save_dataset(dataset, os.path.join(os.path.dirname(__file__), f'datasets/{time.strftime("%H-%M-%S", time.localtime())}.arff'))

video.release()
