import cv2
import dlib
import numpy as np
from math import atan


# This are the points that represent the eyes in the dlib shape_68 model
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

# Value of the image threshold
THRESHOLD = 0


def rotate_image(frame, face_array):
    # Get's the angle of the face based on the extremes of the eyes (bad)
    angle = atan((face_array[36][1]-face_array[45][1])/(face_array[45][0]-face_array[36][0]))*180/np.pi*-0.9

    # Rotate the image
    image_center = tuple(np.array(frame.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)


def face_shape_to_array(face_shape):
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


def colour_eye(frame, eye_points, face_array):
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


def get_eye_rectangle(face_array, eye_points):
    '''
    Gets the rectangles surronding the eyes.
    '''
    # The Xs' values of the eye points
    xs = [face_array[point][0] for point in eye_points]

    # The Ys' values of the eye points
    ys = [face_array[point][1] for point in eye_points]
    
    return [min(xs), max(xs), min(ys), max(ys)]


def contours_pupil(frame, thres, face_array, eye_points):
    '''
    Get's the pupil's centre and draws a circle on it.
    :param frame: image where the circle is drawn.
    :param thres: image where the pupil is extracted.
    :param face_array: Two-dimensional array containing the 68 face points.
    :param eye_points: Points in face_array representing the eye
    '''

    # Get two rectangles that surrounds the eyes
    eye_rectangle = get_eye_rectangle(face_array, eye_points)
    
    # Draws the rectangles
    # cv2.rectangle(frame, (eye_rectangle[0], eye_rectangle[3]), (eye_rectangle[1], eye_rectangle[2]), (0, 255, 0), 1)

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
    except:
        pass


def on_threshold_change(val):
    global THRESHOLD
    THRESHOLD = val

# With the detector, we get faces from frames represented as rectangles 
faces_detector = dlib.get_frontal_face_detector()
# faces_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
# With the predictor, we get the 68 points representing the face from the face_detector's rectangles
face_predictor = dlib.shape_predictor('shape_68.dat')


# Create the threshold window
cv2.namedWindow('Threshold')
cv2.createTrackbar('Value', 'Threshold', 0, 255, on_threshold_change)

# Create the eyes tracker window
cv2.namedWindow('Tracker')

# Kernel is for later dilating the eye zone
kernel = np.ones((9, 9), np.uint8)

# Starts the webcam
video = cv2.VideoCapture(0)
run = True
while(run):
    ret, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get the faces in the frame represented as rectangles
    faces = faces_detector(gray_frame, 1)
    for face in faces:
        # Get the 68 points of the face 
        face_shape = face_predictor(gray_frame, face)
        face_array = face_shape_to_array(face_shape)

        # Rotate image as head tilt
        frame = rotate_image(frame, face_array)

        # Mask is an image sized as frame entirely black except for the eyes (in white)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask = colour_eye(mask, LEFT_EYE_POINTS, face_array)
        mask = colour_eye(mask, RIGHT_EYE_POINTS, face_array)

        # Expand the eye zone
        mask = cv2.dilate(mask, kernel, iterations=5)

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
        contours_pupil(frame, thresh, face_array, LEFT_EYE_POINTS)
        contours_pupil(frame, thresh, face_array, RIGHT_EYE_POINTS)

        # Draw circles on eye points
        # for (x, y) in face_shape[36:48]:
        #         cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # Show the final frames
        cv2.imshow('Tracker', frame)
        cv2.imshow("Threshold", thresh)

        # Pres key 'q' to exit
        if cv2.waitKey(1) == ord('q'):
                run = False
        
video.release()
cv2.destroyAllWindows()