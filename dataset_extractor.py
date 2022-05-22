import cv2
import time
import random

import numpy as np

from lib.pupil_tracker import get_face_parameters, head_tilt, on_threshold_change
from lib.portable_weka_functions import create_dataset, add_to_dataset, save_dataset
from lib.sample_texts import SAMPLE_TEXTS


# Create the threshold window
cv2.namedWindow('Tracker')
cv2.createTrackbar('Threshold', 'Tracker', 0, 255, on_threshold_change)

# Create the dataset for training
dataset = create_dataset()

# Print the text that you should type
print(SAMPLE_TEXTS[random.randint(0,len(SAMPLE_TEXTS)-1)] + '\n\n\n')

# Starts the webcam
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

key = -1
text = ''
while(key != 27):
    ret, frame = video.read()
    if not ret:
        cv2.destroyAllWindows()
        raise Exception('Couldn\'t read from your webcam.')
    
    # Get face parameters
    frame, thresh, face_array, left_pupil, right_pupil = get_face_parameters(frame)

    # Detects the face and a valid key was pressed
    if key not in [-1, 0] and face_array is not None:
        # Delete button
        if key == 8:
            print('\r' + ' ' * len(text), end='', flush=True)
            text = text[:-1]
        # Space button
        elif key == 32:
            print('\r' + ' ' * len(text), end='', flush=True)
            text = ''
        # Uppercase to lowercase
        elif (key>64 and key<91) or key == 209:
            key += 32
            text += chr(key)
        # Add the key to the str
        else: text += chr(key)

        # Get the face parameters and add them to the dataset
        angle = head_tilt(face_array)
        add_to_dataset(dataset, key, left_pupil, right_pupil, face_array[27], angle)
        print('\r' + text, end='', flush=True)
        # print(f'Added: Key {key}, L pupil {str(left_pupil)}, R pupil {str(right_pupil)}, Face position {face_array[27]}, Face angle {angle}')
        
    # Show the final frames
    both = np.vstack((thresh, frame))
    cv2.imshow("Tracker", both)
    
    # Get next key
    key = cv2.waitKey(1)

# Save the dataset
now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
save_dataset(dataset, f'datasets/{now}.arff')

video.release()
cv2.destroyAllWindows()
