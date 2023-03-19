import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 
import pyttsx3
import gtts
import playsound
import pickle
import cv2 as cv

def process_image_data(img):
    rgb_data= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final_image = cv2.flip(rgb_data, 1)
    hand_motion = mp.solutions.hands
    hands = hand_motion.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.7)

    # Results
    output = hands.process(final_image)
    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        data = str(data).strip().split('\n')

        extra = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_extra = []

        for i in data:
            if i not in extra:
                without_extra.append(i)
                        
        clean = []

        for i in without_extra:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])


# load model
with open('svm.pkl', 'rb') as f:
    svm = pickle.load(f)


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera")
    exit()

i = 0    
prev=None
first =True
language ='en'
while True:

    engine = pyttsx3.init()
    ret, frame = cap.read()

    if not ret:
        print("Error. Quiting")
        break
    data = process_image_data(frame)
    
    data = np.array(data)
    y_predicted = svm.predict(data.reshape(-1,63))

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 100)
    fontScale = 3
    color = (255, 0, 0)
    thickness = 5
    

    frame = cv2.putText(frame, str(y_predicted[0]), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    current = y_predicted[0]


    if first:
        prev=current 
        tts = gtts.gTTS(current, lang=language)
        tts.save("hola.mp3")
        playsound.playsound("hola.mp3")
        first=False


    if not first and current != prev:
        prev=current
        tts = gtts.gTTS(current, lang=language)
        tts.save("hola.mp3")
        playsound.playsound("hola.mp3")

cap.release()
cv.destroyAllWindows()