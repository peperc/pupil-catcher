import cv2
import dlib
import os

import numpy as np
from math import atan


# With the detector, we get faces from frames represented as rectangles 
FACES_DETECTOR = dlib.get_frontal_face_detector()

# With the predictor, we get the 68 points representing the face from the face_detector's rectangles
FACE_PREDICTOR = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'models/shape_68.dat'))

# This are the points that represent the eyes in the dlib shape_68 model
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

# Value of the image threshold
THRESHOLD = 0

# Kernel is for later dilating the eye zone
KERNEL = np.ones((9, 9), np.uint8)

# Callback function for threshold change
def on_threshold_change(val):
    '''
    Just a function to call when the user interacts with the threshold trackbar.
    Changes the global value of threshold.
    :param var: Value of the trackbar.
    '''
    
    global THRESHOLD
    THRESHOLD = val


def rotate_image(frame: np.ndarray) -> np.ndarray:
    '''
    Take an image and rotate it until the face is not tilted.
    :param frame: The image with the face that will be rotated.
    :param face_array: The two dimensional array with the 68 face points.
    '''

    # Get the faces in the frame represented as rectangles
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        face = FACES_DETECTOR(gray_frame, 1)[0]
    except:
        return frame
    
    # Get the 68 points of the face 
    face_shape = FACE_PREDICTOR(gray_frame, face)
    face_array = face_shape_to_array(face_shape)
    
    # Get's the angle of the face based on the extremes of the eyes (not so optimal)
    angle = atan((face_array[36][1]-face_array[45][1])/(face_array[45][0]-face_array[36][0]))*180/np.pi*-0.9

    # Rotate the image
    image_center = tuple(np.array(frame.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)


def face_shape_to_array(face_shape) -> np.ndarray:
    '''
    Get a two dimensional array with the coordinates of the 68 points 
    that represents the face. 
    :param face_shape: The object obtained with the 68 points.
    :returns: Array representing the face. 
    '''

    # Initialize the array of coordinades
    face_array = np.zeros((68, 2), dtype=np.int32)
    
    # Loop over the 68 points and convert them to a 2-tuple of coordinates
    for i in range(0, 68):
            face_array[i] = (face_shape.part(i).x, face_shape.part(i).y)

    return face_array


def colour_eye(frame: np.ndarray, eye_points: list[int], face_array: np.ndarray) -> np.ndarray:
    '''
    Colours the shape of the eyes in a certain image.
    :param frame: The image where the eyes should be coloured.
    :param eye_points: Indexes of the points that represent the eye in the.
    :param face_array: Two-dimensional array containing the 68 face points.
    :returns: The modified image.
    '''

    eye_coordinades = [face_array[point] for point in eye_points]
    eye_coordinades = np.array(eye_coordinades, dtype=np.int32)
    # Fills the eye in white
    frame = cv2.fillConvexPoly(frame, eye_coordinades, 255)
    return frame


def get_eye_rectangle(face_array: np.ndarray, eye_points: list[int]) -> list[int]:
    '''
    Gets the rectangles surronding the eyes.
    '''

    # The Xs' values of the eye points
    xs = [face_array[point][0] for point in eye_points]

    # The Ys' values of the eye points
    ys = [face_array[point][1] for point in eye_points]
    
    return [min(xs), max(xs), min(ys), max(ys)]


def contours_pupil(frame: np.ndarray, thres: np.ndarray, face_array: np.ndarray, eye_points: list[int]) -> tuple[int, int]:
    '''
    Get's the pupil's centre and draws a circle on it.
    :param frame: image where the circle is drawn.
    :param thres: image where the pupil is extracted.
    :param face_array: Two-dimensional array containing the 68 face points.
    :param eye_points: Points in face_array representing the eye
    '''

    # Get two rectangles that surrounds the eyes
    eye_rectangle = get_eye_rectangle(face_array, eye_points)
    
    # Draws the rectangle
    cv2.rectangle(frame, (eye_rectangle[0], eye_rectangle[3]), (eye_rectangle[1], eye_rectangle[2]), (0, 255, 0), 1)

    # Finds contours inside the rectangle
    contours, _ = cv2.findContours(thres[eye_rectangle[2]:eye_rectangle[3], eye_rectangle[0]:eye_rectangle[1]], cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    try:
        # Get the biggest contour (the pupil)
        pupil = max(contours, key = cv2.contourArea)
        
        # Get centre of the pupil
        centre = cv2.moments(pupil)
        cx = int(centre['m10']/centre['m00'])
        cy = int(centre['m01']/centre['m00'])

        # Draw the circle
        cv2.circle(frame, (cx + eye_rectangle[0], cy + eye_rectangle[2]), 4, (0, 0, 255), 2)
        
        return (cx, cy)
    except:
        return None


def get_and_draw_pupils(frame: np.ndarray, face_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:

    # Mask is an image sized as frame entirely black except for the eyes (in white)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask = colour_eye(mask, LEFT_EYE_POINTS, face_array)
    mask = colour_eye(mask, RIGHT_EYE_POINTS, face_array)

    # Expand the eye zone
    mask = cv2.dilate(mask, KERNEL, iterations=5)

    # Get the frame with the eyes and the rest in black
    only_eyes_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Get te non-eyes pixels (they are black) and turn them white
    mask = (only_eyes_frame == [0, 0, 0]).all(axis=2)
    only_eyes_frame[mask] = [255, 255, 255]

    # Gets the eyes in grey scale (everything white and the eyes in gray)
    eyes_gray = cv2.cvtColor(only_eyes_frame, cv2.COLOR_BGR2GRAY)

    # Applies threshold to the eyes so the pupils are in black and the rest in white
    _, thresh = cv2.threshold(eyes_gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2) #1
    thresh = cv2.dilate(thresh, None, iterations=4) #2
    thresh = cv2.medianBlur(thresh, 3) #3
    
    # Puts everything in black but the pupils
    thresh = cv2.bitwise_not(thresh)

    # Getting the pupils and contours them
    left_pupil = contours_pupil(frame, thresh, face_array, LEFT_EYE_POINTS)
    right_pupil = contours_pupil(frame, thresh, face_array, RIGHT_EYE_POINTS)

    return frame, thresh, left_pupil, right_pupil


def threshold_finder(frame: np.ndarray) -> int:

    # Get the faces in the frame represented as rectangles
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACES_DETECTOR(gray_frame, 1)
    # Get the 68 points of the face 
    face_shape = FACE_PREDICTOR(gray_frame, faces[0])
    face_array = face_shape_to_array(face_shape)

    # Create the threshold window
    cv2.namedWindow('Threshold')
    cv2.createTrackbar('Value', 'Threshold', 0, 255, on_threshold_change)

    # Create the eyes tracker window
    cv2.namedWindow('Tracker')

    key = -1
    while(key != 27):
        processed = frame.copy()

        processed, thresh, left_pupil, right_pupil = get_and_draw_pupils(processed, face_array)

        # Show the final frames
        cv2.imshow('Tracker', processed)
        cv2.imshow("Threshold", thresh)
        
        # Get next key
        key = cv2.waitKey(1)

    cv2.destroyAllWindows()
    return THRESHOLD
