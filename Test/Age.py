import numpy as np
import cv2 as cv
import dlib
age_model = "D:/PycharmProject/ProjectSource/age_net.caffemodel"
age_txt = "D:/PycharmProject/ProjectSource/deploy_age.prototxt"
gender_model = "D:/PycharmProject/ProjectSource/gender_net.caffemodel"
gender_txt = "D:/PycharmProject/ProjectSource/deploy_gender.prototxt"
haar_xml = "D:/PycharmProject/ProjectSource/haarcascade_frontalface_default.xml"
age = ["0-2","4-6","8-13","15-20","25-32","38-43","48-53","60+"]
detector = dlib.get_frontal_face_detector()
def fatchFace(img):
    ret = detector(img, 1)
    print(ret)
    faceRect = []
    for k, d in enumerate(ret):
        rec = dlib.rectangle(d.left(), d.top(), d.right(), d.bottom())
        cv.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 255, 0), 1)
        faceRect.append([rec.left(),rec.top(),rec.right()-rec.left(),rec.bottom()-rec.top()])
    return faceRect

def predict_age(age_net,img):
    #输入
    blob = cv.dnn.blobFromImage(img,1.0,size=(227,227))
    age_net.setInput(blob,"data")
    #预测模块
    prob = age_net.forward("prob")
    print(prob)
    planes = cv.minMaxLoc(prob)      #返回值为最小值 最大值 最小值位置 最大值位置
    print(planes[3][0])                 #最大值的位置的x坐标
    cv.putText(img,age(planes[3][0]),(2,15),cv.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
def predict_gender(gender_net,img):
    blob = cv.dnn.blobFromImage(img,1.0,size=(227,227))
    gender_net.setInput(blob,"data")
    prob = gender_net.forward("prob")
    # print("prob")
    # print(prob)
    if(prob[0,0]>prob[0,1]):
        gender = "Female"
    else:
        gender = "Male"
    cv.putText(img,gender,(2,25),cv.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
filepath = "D:\GraduationProject\Project1\Image/203.jpg"
src = cv.imread(filepath)
#print(src)
#gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
#cv.namedWindow("input",cv.WINDOW_AUTOSIZ)
#cv.imshow("input",src)
classifier = cv.CascadeClassifier(haar_xml)
color = (0,255,0)
#调用识别人脸
#faceRects = classifier.detectMultiScale(src, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
faceRects = fatchFace(src)
age_net = cv.dnn.readNetFromCaffe(age_txt,age_model)
gender_net = cv.dnn.readNetFromCaffe(gender_txt,gender_model)
#print(len(faceRects))
if len(faceRects):
    for faceRect in faceRects:
        x, y, w, h = faceRect
        #cv.rectangle(src, (x, y), (x + w, y + h), color, 2)
        A = src[y:y + h, x:x + w]
        predict_age(age_net,A)
        predict_gender(gender_net,A)
cv.imshow("predict",src)
cv.waitKey(0)
cv.destroyAllWindows()
