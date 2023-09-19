import cv2
import numpy as np

from PIL import Image

def get_limits(color):
    
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    
    hue = hsvC[0][0][0] 
    
    if hue >= 165:
        lowerLimit = np.array([hue - 10,100,100], dtype=np.uint8)
        upperLimit = np.array([180,255,255], dtype=np.uint8)
    elif hue <= 15:
        lowerLimit = np.array([0,100,100], dtype=np.uint8)
        upperLimit = np.array([hue + 10,255,255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10,100,100], dtype=np.uint8)
        upperLimit = np.array([hue + 10,255,255], dtype=np.uint8)
    return lowerLimit, upperLimit


camera = cv2.VideoCapture(0)
yellow = [0, 255, 255]

while True:
    ret, frame = camera.read()
    
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lowerLimit, upperLimit = get_limits(color = yellow)
   
    mask = cv2.inRange(hsvImg, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)
    
    bbox = mask_.getbbox()
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 0), 3)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWidnows() 