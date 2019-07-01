import cv2
import numpy as np
import dlib
filepath = "D:/PycharmProject/Image/face1.jpg"
img = cv2.imread(filepath)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#人脸识别分类器
classifier = cv2.CascadeClassifier("F:/GraduationProject/opencv-master/data/"
                                   "haarcascades/haarcascade_frontalface_default.xml")
color = (0,255,0)
#调用识别人脸
faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
detector = dlib.get_frontal_face_detector()
ret = detector(img,1)
print(ret)
for k,d in enumerate(ret):
    rec = dlib.rectangle(d.left(), d.top(), d.right(), d.bottom())
    print(rec.left(),rec.top(),rec.right(),rec.bottom())
    cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 255, 0), 1)
    cv2.imshow('img', img)
    cv2.waitKey()
cv2.destroyAllWindows()
# print(img.shape)
# print(len(faceRects))
# if len(faceRects):
#     for faceRect in faceRects:
#         x,y,w,h = faceRect
#         print(faceRect)
#         cv2.rectangle(img,(x,y),(x+h,y+w),color,2)
#         A = gray[y:y + h, x:x + w]
#         #cv2.rectangle(img, faceRect, color, 2)
#         #cv2.circle(img,(x+w//4,y+h//4+30),min(w//8,h//8),color)
#         #cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8),color)
#         #cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4),(x + 5 * w // 8, y + 7 * h // 8), color)
#
#         #B = img[x:x+2*w,y:y+2*h]
#         cv2.imshow("test1",A)
#         #cv2.imshow("test2",B)
#
# cv2.imshow("image",img)
# c = cv2.waitKey(0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()