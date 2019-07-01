import cv2 as cv
import numpy as np
filepath = "D:\GraduationProject\Project1\Image/video2.mp4"
cap = cv.VideoCapture(filepath)
#take the first frame
ret,frame = cap.read()
#r,h,c,w = 250,90,400,125
#r,h,c,w = 250,60,120,70
r,h,c,w = 120,60,250,70
track_window = (c,r,w,h)
roi = frame[r:r+h,c:c+w]
#roi = frame[250:320,120:180]
hsv_roi = cv.cvtColor(roi,cv.COLOR_RGB2HSV)
mask = cv.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10,1)
while(1):
    ret,frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame,cv.COLOR_RGB2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        ret,track_window = cv.meanShift(dst,track_window,term_crit)

        # pts = cv.boxPoints(ret)
        # pts = np.int0(pts)
        # img2 = cv.polylines(frame,[pts],True,255,2)
        x,y,w,h = track_window
        img2 = cv.rectangle(frame,(x,y),(x+w,y+h), 255,2)
        cv.imshow('img2',img2)
        k = cv.waitKey(60) & 0xff   #Esc
        if k == 27:
            break
        else:
            cv.imwrite(chr(k)+".jpg",img2)
    else:
        break
cv.destroyAllWindows()
cap.release()