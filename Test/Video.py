import cv2
import numpy as np
filepath = "D:\GraduationProject\Project1\Image/PersonTest3.mp4"
color = (0,255,0)
cap = cv2.VideoCapture(filepath)
# ret,frame = cap.read()
# cv2.rectangle(frame,(250,120),(320,180),color,2)
# cv2.imshow('frame',frame)
# cv2.waitKey(0)

while(cap.isOpened()):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    cv2.imshow('frame',frame)
    if cv2.waitKey(70) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()