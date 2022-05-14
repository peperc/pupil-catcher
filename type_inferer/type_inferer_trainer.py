import cv2
import dlib
import keyboard

# With the detector, we get faces from frames represented as rectangles 
faces_detector = dlib.get_frontal_face_detector()
# With the predictor, we get the 68 points representing the face from the face_detector's rectangles
face_predictor = dlib.shape_predictor('shape_68.dat')

# Create the eyes tracker window
cv2.namedWindow('Tracker')

# Starts the webcam
video = cv2.VideoCapture(0)
run = True
i = 1
while(run):
    key = keyboard.read_key()
    ret, frame = video.read()

    # Show the final frames
    cv2.imshow('Tracker', frame)
    cv2.waitKey(1)

    if key == 'q':
        run = False
    
cv2.destroyAllWindows()