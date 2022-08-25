import cv2
import time
import random

import numpy as np

from lib.pupil_tracker import get_face_parameters, head_tilt, on_threshold_change, threshold_finder


frame = cv2.imread('secrets/6.jpg')

# Find a optimal threshold
threshold_finder(frame)

# Get face parameters
frame, thresh, face_array, left_pupil, right_pupil = get_face_parameters(frame)

if face_array is not None:
    angle = head_tilt(face_array)