import cv2
import logging

import weka.core.jvm as jvm

from lib.pupil_tracker import get_face_parameters, head_tilt, threshold_finder
from lib.weka_functions import create_dataset, add_to_dataset, save_dataset


VIDEO_PATH = 'shots/4.mp4'

# Gets the video and it's frames count
video = cv2.VideoCapture(VIDEO_PATH)
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
        # Get face parameters
        frame, thresh, face_array, left_pupil, right_pupil = get_face_parameters(frame)

        if face_array is not None:
            angle = head_tilt(face_array)
            add_to_dataset(dataset, None, left_pupil, right_pupil, face_array[27], angle)
            # print(f'Added: Key {None}, L pupil {str(left_pupil)}, R pupil {str(right_pupil)}, Face position {face_array[27]}, Face angle {angle}')


        print(f'Processed frame {i} out of {total_frames}')
        i += 1

# Save the dataset
video_name = VIDEO_PATH.split('/')[-1]
save_dataset(dataset, f'{video_name}.arff')
