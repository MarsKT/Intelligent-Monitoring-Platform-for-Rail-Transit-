import cv2
import math
import numpy as np
import sys
import kcfddst.tracker as kcftracker
def center(box):
    return [box[0]+box[2]/2,box[1]+box[3]/2]
def getInstance(p1,p2):
    distance = math.pow((p1[0]-p2[0]),2)+pow((p1[1]-p2[1]),2)
    distance = math.sqrt(distance)
    return  distance
def cutRoi(roi,i):
    roi[2] -= i
    roi[3] -= i
# tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE']
# # tracker_type = tracker_types[6]
# # if tracker_type == 'BOOSTING':
# #     tracker = cv2.TrackerBoosting_create()
# # if tracker_type == 'MIL':
# #     tracker = cv2.TrackerMIL_create()
# # if tracker_type == 'KCF':
# #     tracker = cv2.TrackerKCF_create()
# # if tracker_type == 'TLD':
# #     tracker = cv2.TrackerTLD_create()
# # if tracker_type == 'MEDIANFLOW':
# #     tracker = cv2.TrackerMedianFlow_create()
# # if tracker_type == 'GOTURN':
# #     tracker = cv2.TrackerGOTURN_create()
# # if tracker_type == 'MOSSE':
# #     tracker = cv2.TrackerMOSSE_create()

#frame = 0
filepath = 'D:\GraduationProject\Project1\Image/MOT17-02.mp4'
cap = cv2.VideoCapture(filepath)
_,frame = cap.read()
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
hog = cv2.HOGDescriptor()
detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
hog.setSVMDetector(detector)

found,wight = hog.detectMultiScale(gray)
trackerArray = []
for i in found:
    tracker = kcftracker.KCFTracker(True, True, True)
    tracker.init(i, frame)
    trackerArray.append(tracker)



#print(found[2])
# found[2][0] -= 0
# found[2][1] += 30
# found[2][3] -= 60
# found[2][2] -= 30
#tracker.init(frame,tuple(found[1]))
framecount = 0
while(cap.isOpened()):
    _ret, frame = cap.read()
    if _ret == False:
        break
    framecount += 1
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if False: #framecount%5 == 0:
        f,w = hog.detectMultiScale(gray)
        minL = 100000
        index = 0
        for i in range(len(f)):
            l = getInstance( center(box), center(f[i]) )
            print('distance %d: %f'%(i,l))
            if l < minL:
                minL = l
                index = i
        tracker.init(frame,tuple(f[i]))
        p1 = (int(f[i][0]),int(f[i][1]))
        p2 = (int(f[i][0] + f[i][2]), int(f[i][1] + f[i][3]))
        ok,box = tracker.update(frame)
        cv2.rectangle(frame,p1,p2,(255,0,0),2,1)
    else:
    #cv2.equalizeHist(gray,gray)
        for i in trackerArray:
            box = i.update(frame)
            # for box in boxes:
            #print(box)
            p1 = (int(box[0]),int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame,p1,p2,(255,0,0),2,1)

    cv2.imshow('T',frame)
    k = cv2.waitKey(20) & 0xff
    if k == 27: break
cap.release()
cv2.destroyAllWindows()
