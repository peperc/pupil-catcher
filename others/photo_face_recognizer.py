import cv2
import dlib

import numpy as np


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


img = cv2.imread('shots/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale 
detector = dlib.get_frontal_face_detector()
# detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
rects = detector(gray, 1) # rects contains all the faces detected

predictor = dlib.shape_predictor('face_recognizer/shape_68.dat')
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    for (x, y) in shape:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

# Save the image
cv2.imwrite('shots/1_modified.jpg', img)