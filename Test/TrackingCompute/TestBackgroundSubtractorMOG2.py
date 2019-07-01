import numpy as np


#检测过后 发现 首先应当降低图片分辨率 提高一点处理速度，然后就是手部抖动问题。
import cv2

class Position(object):
    def __init__(self,_x,_y,_w,_h):
        self.x = _x
        self.y = _y
        self.w = _w
        self.h = _h
    def x1(self):
        return self.x
    def y1(self):
        return self.y
    def w1(self):
        return self.w
    def h1(self):
        return self.h

#filepath = 'D:\GraduationProject\Project1\Image/MOT17-02.mp4'
filepath = 'D:\GraduationProject\Project1\Image/passageway1.avi'
cap = cv2.VideoCapture(filepath)

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
fgbgK = cv2.createBackgroundSubtractorKNN()

midHeight = int(cap.get(4) / 2)
maxWidth = cap.get(3)
maxHeight = cap.get(4)
boundaryPt1 = [maxWidth-60,0]
boundaryPt2 = [maxWidth-50,maxHeight]

while(1):
    ret, frame = cap.read()
    cv2.line(frame, (int(boundaryPt1[0]), int(boundaryPt1[1])),(int(boundaryPt2[0]), int(boundaryPt2[1])), (0, 255, 0), 1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    blur = cv2.medianBlur(fgmask, 5)

    thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]  # 高斯背景法中 阴影的色值为 grey = 127
    #thresh1 = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]
    # 提取边缘信息
    sobelx = cv2.Sobel(thresh,cv2.CV_32F,1,0,ksize = 3)
    sobely = cv2.Sobel(thresh,cv2.CV_32F,0,1,ksize = 3)
    sobelDst = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    #sobelDst = cv2.medianBlur(sobelDst, 5)
    sobelDst = cv2.convertScaleAbs(sobelDst)

    # print('sobelDst:',end=' ')
    # print(sobelDst)
    #定义结构元素，然后进行 开闭运算  获取轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #闭运算填充内部孔洞
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #开运算去除噪点
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # print('opening:',end=' ')
    # print(opening)
    Dst = cv2.bitwise_or(opening,sobelDst)
    # 此处获取外部轮廓
    contours = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    now = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500 or area > 20000:
            continue
        else:
            (x, y, w, h) = cv2.boundingRect(c)
            now.append(Position(x,y,w,h))
    for p in now:
        cv2.rectangle(frame,(p.x1(),p.y1()),(p.x1()+p.w1(),p.y1()+p.h1()),(0,0,255),1)

    mask_opening = cv2.inRange(Dst, np.array([0]), np.array([128]))

    #截取mask_opening区域的图像
    #noBg = cv2.bitwise_and(frame, frame, mask=mask_opening)
    #cv2.drawContours(frame,now,-1,(0,0,255),3)



    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.namedWindow('DST', cv2.WINDOW_NORMAL)
    #cv2.imshow('frame',thresh)
    cv2.imshow('DST', Dst)
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()