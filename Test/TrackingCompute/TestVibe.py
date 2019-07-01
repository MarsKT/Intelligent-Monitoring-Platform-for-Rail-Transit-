import numpy as np
from Vibe import *

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

filepath = 'D:\GraduationProject\Project1\Image/passageway1.avi'
#filepath = 'D:\GraduationProject\Project1\Image/re.mp4'
cap = cv2.VideoCapture(filepath)
count = 0
while(1):
    count += 1
    print(count)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #fgmask = fgbgK.apply(gray)
    if count == 1:
        v = Vibe(gray)
        #v.__init__(gray)
        print('vibe init complete')
    else:
        foregroud = v.update(gray)
        # blur = cv2.medianBlur(foregroud, 5)
        # #thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]  # 高斯背景法中 阴影的色值为 grey = 127
        # #定义结构元素，然后进行 开闭运算  获取轮廓
        # sobelx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        # sobely = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        # sobelDst = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        # sobelDst = cv2.convertScaleAbs(sobelDst)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # #闭运算填充内部孔洞
        # #closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
        # #开运算去除噪点
        # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        # Dst = cv2.bitwise_or(opening, sobelDst)
        #
        # mask_opening = cv2.inRange(opening, np.array([0]), np.array([128]))

    #截取mask_opening区域的图像
    #noBg = cv2.bitwise_and(frame, frame, mask=mask_opening)
    #cv2.drawContours(frame,now,-1,(0,0,255),3)



        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.namedWindow('DST', cv2.WINDOW_NORMAL)
        cv2.imshow('frame',frame)
        cv2.imshow('DST',foregroud)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()