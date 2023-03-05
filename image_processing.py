from image_func import *
from time import sleep
import os



Processed_Path = "D:\MIniProject\Processed_Data" 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    

    # Loop sign signs
    for sign in signs:
        # Loop through image
        for sequence in range(0,1200):
                #reading image
                frame=cv2.imread('D:\\MIniProject\\Raw_Data\\{}\\{}.jpg'.format(sign,sequence))
                # Make detections
                image, results = mediapipe_detection(frame, hands)
                # Draw landmarks
                draw_styled_landmarks(image, results)

                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(Processed_Path, sign, str(sequence))
                np.save(npy_path, keypoints)


