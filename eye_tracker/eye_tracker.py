import cv2
import dlib
import numpy as np


# This are the points that represent the eyes in the dlib shape_68 model
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]


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
    # eye_coordinades = np.array(eye_coordinades, dtype=np.int32)
    # Fills the eye in white
    frame = cv2.fillConvexPoly(frame, eye_coordinades, 255)
    return frame


def contouring(partial_frame, eye_centre, frame, right=False):
    '''
    Finds the biggest countour in the partial frame and draws a circle in the centre of it 
    in the frame.
    '''

    # Find the contours
    contours, _ = cv2.findContours(partial_frame, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        # Get the biggest contour (the pupil)
        pupil = max(contours, key = cv2.contourArea)

        # Get centre coordinades
        centre = cv2.moments(pupil)
        cx = int(centre['m10']/centre['m00'])
        cy = int(centre['m01']/centre['m00'])

        # Draw the circle
        if right:
            cx += eye_centre
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass


# With the detector, we get faces from frames represented as rectangles 
faces_detector = dlib.get_frontal_face_detector()
# With the predictor, we get the 68 points representing the face from the face_detector's rectangles
face_predictor = dlib.shape_predictor('shape_68.dat')

# Create the threshold window
cv2.namedWindow('Threshold')
cv2.createTrackbar('Value', 'Threshold', 0, 255, None)

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

        # Mask is an image sized as frame entirely black except for the eyes (in white)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask = colour_eye(mask, LEFT_EYE_POINTS, face_array)
        mask = colour_eye(mask, RIGHT_EYE_POINTS, face_array)

        # Expand the eye zone
        mask = cv2.dilate(mask, kernel, iterattions=5)

        # Get the frame with the eyes and the rest in black
        only_eyes_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Get te non-eyes pixels (they are black) and turn them white
        mask = (only_eyes_frame == [0, 0, 0]).all(axis=2)
        only_eyes_frame[mask] = [255, 255, 255]

        # Gets the eyes in grey scale (everything white and the eyes in gray)
        eyes_gray = cv2.cvtColor(only_eyes_frame, cv2.COLOR_BGR2GRAY)

        # Applies threshold to the eyes so the pupils are in black
        threshold = cv2.getTrackbarPos('Value', 'Threshold')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        
        # Puts everything in black but the pupils
        thresh = cv2.bitwise_not(thresh)

        # Getting the centre between the eyes and contoour the pupils
        eye_centre = (face_array[42][0] + face_array[39][0]) // 2
        # Contours left pupil
        contouring(thresh[:, :eye_centre], eye_centre, frame)
        # Contours right pupil
        contouring(thresh[:, eye_centre:], eye_centre, frame, True)

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