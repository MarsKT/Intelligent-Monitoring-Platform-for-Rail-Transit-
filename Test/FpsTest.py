import cv2
import numpy as np
import dlib as db
import time
dt = db.get_frontal_face_detector()
haar_xml = "D:/PycharmProject/ProjectSource/haarcascade_frontalface_default.xml"
filepath = "D:\GraduationProject\Project1\Image/facedetection.mp4"
#src = cv2.imread(filepath)
cap = cv2.VideoCapture(filepath)
classifier = cv2.CascadeClassifier(haar_xml)
def fatchFace(img):
    ret = dt(img, 1)
    faceRect = []
    for k, d in enumerate(ret):
        rec = db.rectangle(d.left(), d.top(), d.right(), d.bottom())
        #cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 255, 0), 1)
        #faceRect.append([rec.left(),rec.top(),rec.right()-rec.left(),rec.bottom()-rec.top()])
        faceRect.append([rec.left(), rec.top(), rec.right()-rec.left(), rec.bottom()-rec.top()])
    return faceRect
color = (0,255,0)
framecount = 0
fpscount = 0
while cap.isOpened():
    framecount += 1
    start = time.time()
    ret,frame = cap.read()
    if not ret:
        break
    if framecount == 100:
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #faceret = fatchFace(gray)
    faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    end = time.time()
    second = end-start
    fps = 1/second
    fpscount += fps
print(fpscount/framecount)
