import cv2
import numpy as np

filepath = 'D:\GraduationProject\Project1\Image/MOT17-02.mp4'
cap = cv2.VideoCapture(filepath)
_,frame = cap.read()
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
hog = cv2.HOGDescriptor()
detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
hog.setSVMDetector(detector)

found,wight = hog.detectMultiScale(gray)
index = 2
track_window = tuple(found[index])
print(track_window)
roi = frame[found[index][1]:found[index][1]+found[index][3],found[index][0]:found[index][0]+found[index][2]]
print(found[index])
hsv_roi = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)


term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)
while cap.isOpened():
    ret,frame = cap.read()
    p1 = (int(found[index][0]), int(found[index][1]))
    p2 = (int(found[index][0] + found[index][2]), int(found[index][1] + found[index][3]))
    cv2.rectangle(frame,p1,p2,(255,0,0),2,1)
    if ret == True:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        ret,track_window = cv2.CamShift(dst,track_window,term_crit)

        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img = cv2.polylines(frame,[pts],True,255,2)
        cv2.imshow('t',img)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
cap.release()
cv2.destroyAllWindows()
