import cv2
import math
import numpy as np
import sys
def center(box):
    return [box[0]+box[2]/2,box[1]+box[3]/2]
def getInstance(p1,p2):
    distance = math.pow((p1[0]-p2[0]),2)+pow((p1[1]-p2[1]),2)
    distance = math.sqrt(distance)
    return  distance
def cutRoi(roi):
    ret = [[roi[0]-50],[roi[1]-50],[roi[2]-50],[roi[3]-50],]
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
def isIn(point):
    if point[0]>=x and point[0] <=x+w:
        if point[1] >=y and point[1] <= y+h:
            return True
    return False

def checkTouchSide(point):
    if (point[0]<=x and point[0] >= x-tolorance):
        if(point[1] >= y-tolorance and point[1] <= y+tolorance+h):
            return True
    elif (point[0]>=x+w and point[0] <= x+w+tolorance):
        if (point[1] >= y - tolorance and point[1] <= y + tolorance + h):
            return True
    return False

#frame = 0
filepath = 'D:\GraduationProject\Project1\Image/MOT17-02.mp4'
cap = cv2.VideoCapture(filepath)
maxH = cap.get(4)
maxW = cap.get(3)
x = int(maxW/2-50)
y = int(maxH/2-100)
w = 200
h = 300
tolorance = 10

hog = cv2.HOGDescriptor()
detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
hog.setSVMDetector(detector)

# for i in found:
#     tracker.add(cv2.TrackerKCF_create(),gray,tuple(i))
trackerArray = []
resultArray = []
frameCount = 0
while cap.isOpened():
    # print('framecount = ',end='  ')
    # print(frameCount)
    ret, frame = cap.read()
    if not ret:
        break
    frameCount += 1
    if frameCount%10 == 0:
        frameCount = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, wight = hog.detectMultiScale(gray)
        for i in found:
            if checkTouchSide(center(i)):
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame,tuple(i))
                trackerArray.append(tracker)
                p1 = (int(i[0]), int(i[1]))
                p2 = (int(i[0] + i[2]), int(i[1] + i[3]))
                cv2.rectangle(frame,p1,p2,(0,0,255),1)
        print('trackerArray: ',end='  ')
        print(len(trackerArray))
    else:
        for i in reversed(trackerArray):
            _, box = i.update(frame)

            if isIn(center(box)) or checkTouchSide(center(box)):
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 1)
            else:
                resultArray.append(box)
                trackerArray.remove(i)
                resultArray.append(box)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imshow('t', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27: break
print("resultArray: " ,end='  ')
print(len(resultArray))
cap.release()
cv2.destroyAllWindows()

