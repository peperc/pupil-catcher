import os
import cv2
import dlib

from lib.pupil_tracker import get_and_draw_pupils, face_shape_to_array, on_threshold_change


# With the detector, we get faces from frames represented as rectangles 
faces_detector = dlib.get_frontal_face_detector()
# With the predictor, we get the 68 points representing the face from the face_detector's rectangles
face_predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'lib/models/shape_68.dat'))

# Create the threshold window
cv2.namedWindow('Threshold')
cv2.createTrackbar('Value', 'Threshold', 0, 255, on_threshold_change)

# Create the eyes tracker window
cv2.namedWindow('Tracker')

# Starts the webcam
video = cv2.VideoCapture(0)

pressed_key = 0
while(pressed_key != 27):
    ret, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get the faces in the frame represented as rectangles
    faces = faces_detector(gray_frame, 1)
    for face in faces:
        # Get the 68 points of the face 
        face_shape = face_predictor(gray_frame, face)
        face_array = face_shape_to_array(face_shape)

        frame, thresh, left_pupil, right_pupil = get_and_draw_pupils(frame, face_array)

        # Show the final frames
        cv2.imshow('Tracker', frame)
        cv2.imshow("Threshold", thresh)

        # Pres key 'ESC' to exit
        pressed_key =  cv2.waitKey(1)
        
video.release()
cv2.destroyAllWindows()