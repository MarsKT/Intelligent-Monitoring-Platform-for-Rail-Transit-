import cv2

filepath = "D:/PycharmProject/Image/face1.jpg"
img = cv2.imread(filepath)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(img)
print(gray.shape)
#人脸识别分类器
classifier = cv2.CascadeClassifier("F:/GraduationProject/opencv-master/data/"
                                   "haarcascades/haarcascade_frontalface_default.xml")
color = (0,255,0)
#调用识别人脸
faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
print(len(faceRects))
if len(faceRects):
    for faceRect in faceRects:
        x,y,w,h = faceRect
        #框人脸
        cv2.rectangle(img,(x,y),(x+h,y+w),color,2)
        #cv2.rectangle(img, faceRect, color, 2)
        cv2.circle(img,(x+w//4,y+h//4+30),min(w//8,h//8),color)
        cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8),color)
        cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4),(x + 5 * w // 8, y + 7 * h // 8), color)
cv2.imshow("image",img)
c = cv2.waitKey(10)
cv2.waitKey(0)
cv2.destroyAllWindows()