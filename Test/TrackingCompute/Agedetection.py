import cv2
import dlib as db
import numpy as np
import tkinter as tk
age_model = "D:/PycharmProject/ProjectSource/age_net.caffemodel"
age_txt = "D:/PycharmProject/ProjectSource/deploy_age.prototxt"
gender_model = "D:/PycharmProject/ProjectSource/gender_net.caffemodel"
gender_txt = "D:/PycharmProject/ProjectSource/deploy_gender.prototxt"
hogFeature = 'D:\GraduationProject\Project1\Test/TrackingCompute/myHOgDector'
filepath = 'D:\GraduationProject\Project1\Image/facedetection.mp4'
age_net = cv2.dnn.readNetFromCaffe(age_txt,age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_txt,gender_model)

age = ["0-2","4-6","8-13","15-20","25-32","38-43","48-53","60+"]
agenum = [0,0,0,0,0,0,0,0]
gender = [0,0]

cap = cv2.VideoCapture(filepath)

tracker = cv2.TrackerKCF_create()   #创建追踪器
trackerList = []
posList = []
dt = db.get_frontal_face_detector()
tolerance = 40
#界面
root = tk.Tk()
root.title('年龄性别构成')
text1 = tk.StringVar()
text2 = tk.StringVar()
text3 = tk.StringVar()
text4 = tk.StringVar()
text5 = tk.StringVar()
text6 = tk.StringVar()
text7 = tk.StringVar()
text8 = tk.StringVar()
text9 = tk.StringVar()
text10 = tk.StringVar()
label1 = tk.Label(root,text='0-2岁：').grid(row=0, column=0)
label2 = tk.Label(root,text='2-4岁：').grid(row=1, column=0)
label3 = tk.Label(root,text='8-13岁：').grid(row=2, column=0)
label4 = tk.Label(root,text='15-20岁：').grid(row=3, column=0)
label5 = tk.Label(root,text='25-32岁：').grid(row=4, column=0)
label6 = tk.Label(root,text='38-43岁：').grid(row=5, column=0)
label7 = tk.Label(root,text='48-53岁：').grid(row=6, column=0)
label8 = tk.Label(root,text='60+岁：').grid(row=7, column=0)
label9 = tk.Label(root,text='男：').grid(row=8, column=0)
label10 = tk.Label(root,text='女：').grid(row=9, column=0)

label11 = tk.Label(root,textvariable=text1).grid(row=0, column=1)
label22 = tk.Label(root,textvariable=text2).grid(row=1, column=1)
label33 = tk.Label(root,textvariable=text3).grid(row=2, column=1)
label44 = tk.Label(root,textvariable=text4).grid(row=3, column=1)
label55 = tk.Label(root,textvariable=text5).grid(row=4, column=1)
label66 = tk.Label(root,textvariable=text6).grid(row=5, column=1)
label77 = tk.Label(root,textvariable=text7).grid(row=6, column=1)
label88 = tk.Label(root,textvariable=text8).grid(row=7, column=1)
label99 = tk.Label(root,textvariable=text9).grid(row=8, column=1)
label1010 = tk.Label(root,textvariable=text10).grid(row=9, column=1)

def predict_age(age_net,img):
    #输入
    #print(img)
    blob = cv2.dnn.blobFromImage(img,1.0,size=(227,227))
    age_net.setInput(blob,"data")
    #预测模块
    prob = age_net.forward("prob")
    #print(prob)
    planes = cv2.minMaxLoc(prob)      #返回值为最小值 最大值 最小值位置 最大值位置
    #print(planes[3][0])                 #最大值的位置的x坐标
    #cv2.putText(img,age[planes[3][0]],(2,15),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
    return planes[3][0]-1
def predict_gender(gender_net,img):
    #print(img)
    blob = cv2.dnn.blobFromImage(img,1.0,size=(227,227))
    gender_net.setInput(blob,"data")
    prob = gender_net.forward("prob")
    # print("prob")
    # print(prob)
    if(prob[0,0]>prob[0,1]):
        gender = "Female"
        return 0
    else:
        gender = "Male"
        return 1
    #cv2.putText(img,gender,(2,25),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

def center(box):
    return [box[0]+box[2]/2,box[1]+box[3]/2]

def fatchFace(img):
    ret = detector(img, 1)
    faceRect = []
    for k, d in enumerate(ret):
        rec = db.rectangle(d.left(), d.top(), d.right(), d.bottom())
        #cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 255, 0), 1)
        #faceRect.append([rec.left(),rec.top(),rec.right()-rec.left(),rec.bottom()-rec.top()])
        faceRect.append([rec.left(), rec.top(), rec.right()-rec.left(), rec.bottom()-rec.top()])
    return faceRect

def getDetector(filepath):
    hog = cv2.HOGDescriptor()
    hog.load(filepath)
    return dt

midHeight = int(cap.get(4) / 2)
maxWidth = cap.get(3)
maxHeight = cap.get(4)
midWidth = maxWidth//2
# pt1 = [maxWidth/2,0]
# pt2 = [maxWidth/2,maxHeight]
color = (0,255,0)
num = 0

#主程序
detector = getDetector(hogFeature)
frameCount = 1
peopleNum = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frameCount += 1
    if frameCount%5 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i in trackerList:
            ok,box = i.update(frame)
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, (0, 0, 255), 1)
            posList.append(center(box))
        #cv2.line(frame,(int(pt1[0]),int(pt1[1])),(int(pt2[0]),int(pt2[1])),color,1)
        #area = gray[int(midWidth):int(maxWidth),int(midHeight):int(maxHeight)]
        faceret = fatchFace(gray)
        if len(faceret):
            for i in faceret:
                flag = False
                p1 = (int(i[0]), int(i[1]))
                p2 = (int(i[0] + i[2]), int(i[1] + i[3]))
                cv2.rectangle(frame,p1,p2,(255,0,0),1)
                centerpos = center(i)
                for j in posList:
                    if abs(centerpos[0]-j[0])<=tolerance:
                        flag = True
                        break
                if flag == False:
                    print('new face')
                    #print(i)
                    A = frame[i[1]:i[1] + i[3], i[0]:i[0] + i[2]]
                    if len(A):
                        ageIndex = predict_age(age_net,A)
                        agenum[ageIndex] += 1
                        genderIndex = predict_gender(gender_net,A)
                        gender[genderIndex] += 1
                        peopleNum += 1
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame,tuple(i))
                        trackerList.append(tracker)
    else:
        for i in trackerList:
            #print('更新')
            ok,box = i.update(frame)
            if ok:
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 1)
            # else:
            #     print('跟踪失败')
    cv2.imshow('frame',frame)
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
text1.set(agenum[0])
text2.set(agenum[1])
text3.set(agenum[2])
text4.set(agenum[3])
text5.set(agenum[4])
text6.set(agenum[5])
text7.set(agenum[6])
text8.set(agenum[7])
text9.set(gender[0])
text10.set(gender[1])
print(peopleNum)
print(agenum)
print(gender)
root.mainloop()


