import numpy as np
import cv2 as cv
from pathlib import Path
import time

def get_image():
    Class = 'you'
    Path('DATASET/'+Class).mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    i = 0    
    while True:
       
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # frame = cv.flip(frame,1)
        i+= 1
        if i % 5==0:
            cv.imwrite('DATASET/'+Class+'/'+str(i)+'.png',frame)
        time.sleep(0.1)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q') or i > 100:
            break
  
    cap.release()
    cv.destroyAllWindows()
if __name__ == "__main__":
   get_image()
  