import numpy as np
import cv2
import colorsys
import collections

def AverFiltering(src):
    dst = src
    h = src.shape[0]
    w = src.shape[1]
    for j in range(h):
        if j!=0 :
            for i in range(2):
                if i!= 0 :
                    if (i-1>=0)&(j-1>=0)&(i+1<w)&(j+1<h):
                        dst[j][i][0] = (src[j][i][0] + src[j-1][i-1][0] + src[j-1][i][0]
                                        + src[j][i-1][0] + src[j-1][i+1][0] + src[j+1][i-1][0]
                                        + src[j+1][i+1][0] + src[j][i+1][0] + src[j+1][i][0]) / 9
                        dst[j][i][1] = (src[j][i][1] + src[j - 1][i - 1][1] + src[j - 1][i][1]
                                        + src[j][i - 1][1] + src[j - 1][i + 1][1] + src[j + 1][i - 1][1]
                                        + src[j + 1][i + 1][1] + src[j][i + 1][1] + src[j + 1][i][1]) / 9
                        dst[j][i][2] = (src[j][i][2] + src[j - 1][i - 1][2] + src[j - 1][i][2]
                                        + src[j][i - 1][2] + src[j - 1][i + 1][2] + src[j + 1][i - 1][2]
                                        + src[j + 1][i + 1][2] + src[j][i + 1][2] + src[j + 1][i][2]) / 9
                    else:
                        dst[j][i][0] = src[j][i][0]
                        dst[j][i][1] = src[j][i][1]
                        dst[j][i][2] = src[j][i][2]
    return dst
def MidFiltering(src):
    dst = src
    h = src.shape[0]
    w = src.shape[1]
    for j in range(h):
        if j!=0 :
            for i in range(2):
                All = []
                if i!= 0 :
                    if (i-1>0)&(j-1>0)&(i+1<w)&(j+1<h):
                        ret = All + [ src[j][i][0] , src[j-1][i-1][0] , src[j-1][i][0]
                                        , src[j][i-1][0] , src[j-1][i+1][0] , src[j+1][i-1][0]
                                        , src[j+1][i+1][0] , src[j][i+1][0] , src[j+1][i][0] ]
                        result = ret.sort()[len(ret)/2]
                        dst[j][i][0] = result
                        All.clear()
                        ret = All + [ src[j][i][1] , src[j-1][i-1][1] , src[j-1][i][1]
                                        , src[j][i-1][1] , src[j-1][i+1][1] , src[j+1][i-1][1]
                                        , src[j+1][i+1][1] , src[j][i+1][1] , src[j+1][i][1] ]
                        result = ret.sort()[len(ret)/2]
                        dst[j][i][1] = result
                        All.clear()
                        ret = All + [ src[j][i][2] , src[j-1][i-1][2] , src[j-1][i][2]
                                        , src[j][i-1][2] , src[j-1][i+1][2] , src[j+1][i-1][2]
                                        , src[j+1][i+1][2] , src[j][i+1][2] , src[j+1][i][2] ]
                        result = ret.sort()[len(ret)/2]
                        dst[j][i][2] = result
                        All.clear()
                    else:
                        dst[j][i] = src[j][i]
    return dst
def Sobel_Contour(img):
    h = img.shape[0]
    w = img.shape[1]
    res = np.zeros((w,h))
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, 1], [0, 0, 0], [1, 2, -1]]
    for x in range(1,w-1):
        for y in range(1,h-1):
            sub = [[img[x-1][y-1],img[x-1][y],img[x-1][y+1]],
                   [img[x][y-1]],img[x][y],img[x][y+1],
                   img[x+1][y-1],img[x+1][y],img[x+1][y+1]]
            sub = np.array(sub)
            var_x = sum(sum(sub * sobel_x))
            var_y = sum(sum(sub * sobel_y))
            var = abs(var_x)+abs(var_y)
            res[x][y] = var
    return res

#Position Class  -------------------------------------------------
class Position(object):
    def __init__(self,_x,_y,_w,_h):
        self.x = _x
        self.y = _y
        self.w = _w
        self.h = _h
    def xp(self):
        return self.x
    def yp(self):
        return self.y
    def wp(self):
        return self.w
    def hp(self):
        return self.h

#People Class  -----------------------------------------------------
class People(object):
    def __init__(self,_x,_y,_w,_h,_roi,_hue):
        self.x = _x
        self.y = _y
        self.w = _w
        self.h = _h
        self.roi = _roi

        #Display of the contour while tracking

        self.hue = _hue
        self.color = hsv2rgb(self.hue%1,1,1)

        #Motion Descriptors
        self.center = [_x + _w/2 ,_y + _h/2]
        self.isIn = checkPosition(boundaryPt1, boundaryPt2, self.center, inCriterion)
        self.isInChangeFrameCount = toleranceCountIOStatus
        self.speed = [0,0]
        self.missingCount = 0

        #Roi - Region of Interest
        self.maxRoi = _roi
        self.roi = _roi


    def xp(self):
        return self.x

    def yp(self):
        return self.y

    def wp(self):
        return self.w

    def hp(self):
        return self.h

    def roip(self):
        return self.roi

    def colorp(self):
        return self.color

    def centerp(self):
        return self.center

    def maxRoip(self):
        return self.maxRoi

    def isInp(self):
        return self.isIn

    def speedp(self):
        return self.speed

    def missingCountp(self):
        return self.missingCount

    def isInChangeFrameCountp(self):
        return self.isInChangeFrameCount

    def set(self,name,value):
        if name == 'x':
            self.x = value
        elif name == 'y':
            self.y = value
        elif name == 'w':
            self.w = value
        elif name == 'h':
            self.h = value
        elif name == 'center':
            self.center = value
        elif name == 'roi':
            self.roi = value
            if self.roi.shape[0] * self.roi.shape[1] > self.maxRoi.shape[0] * self.maxRoi.shape[1]:
                self.maxRoi = self.roi
        elif name == "speed":
            self.speed = value
        elif name == "missingCount":
            self.missingCount = value
        elif name == "isIn":
            self.isIn = value
        elif name == "isInChangeFrameCount":
            self.isInChangeFrameCount = value
        else:
            return

#--------------------------------- 所需要的函数 ------------------------------------------------
#----------------------------------------------------------------------------------------------

#计算平均区域大小

def averageSize():
    sum = 0
    for i in humanSizeSample:
        sum += i
    return sum/sampleSize

#关注顶部和底部
def checkTouchSide(x,y,w,h,maxW,maxH,tolerance):
    if x <= 0:
        return True
    elif y-tolerance <= 0:
        return True
    elif x+w >= maxW:
        return True
    elif y+h+tolerance >= maxH:
        return True
    else:
        return False

#获取最外部轮廓
def getExteriorRect(pts):
    xArray = []
    yArray = []
    for pt in pts:
        xArray.append(pt[0])
        yArray.append(pt[1])
    xArray = sorted(xArray)
    yArray = sorted(yArray)
    return (xArray[0],yArray[0],xArray[3]-xArray[0],yArray[3]-yArray[0])

#将hsv转换为rgb--HSV也就是HSB->Hue-色相 Saturation-饱和度 Value(brightness)-明度
def hsv2rgb(h,s,v):
    return tuple(i*255 for i in colorsys.hsv_to_rgb(h,s,v))

#判断目标点在直线两侧的哪一侧
def checkPosition(boundaryPt1,boundaryPt2,curPos,inCriterion):
    # 直线方程斜率
    m = (boundaryPt2[1] - boundaryPt1[1]) / (boundaryPt2[0] - boundaryPt1[0])
    # 直线方程截距
    c = boundaryPt2[1] - m * boundaryPt2[0]
    if inCriterion == "<":
        if curPos[0] * m + c > curPos[1]:
            return True
        else:
            return False
    elif inCriterion == ">":
        if curPos[0] * m + c > curPos[1]:
            return True
        else:
            return False
    else:
        return False

def nothing(x):
    pass

#---------------------------视 频 资 源----------------------------
#------------------------------------------------------------------
#视频所在位置
filepath = 'D:\GraduationProject\Project1\Image/passageway1.avi'
srcWebcam = 0
srcMain = ''
cap = cv2.VideoCapture(filepath)


#------------------------Start Detect and Tracking-----------------
#------------------------------------------------------------------
minArea = 500               #默认最小人体区域
maxArea = 20000             #默认最大人体区域
noFrameToCollectSample = 100
toleranceRange = 50         # use for error calculation
toleranceCount = 10         # maxinux number if fram an object need to present in order to be accepted
toleranceCountIOStatus = 3  # 规定在保持In/Out状态帧数后才可以判定出入
startHue = 0                # In HSV this is RED
hueIncrementValue = 0.1     # increment color every time to differentiate between different people


#-----------------------------setup---------------------------------
#-------------------------------------------------------------------

# fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold = 16, detectShadows=True)

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
fgbgK = cv2.createBackgroundSubtractorKNN()

#检测运动物体队列大小
sampleSize = 100
humanSizeSample = collections.deque(maxlen=sampleSize)

#VideoCapture.get() 0:视频播放当前位置，毫秒为单位  1:帧索引  2:视频播放相对位置  3:帧的宽度  4:帧的高度
#                   5:帧速率
midHeight = int(cap.get(4) / 2)
maxWidth = cap.get(3)
maxHeight = cap.get(4)

inCriterion = '<'
boundaryPt1 = [maxWidth-60,0]
boundaryPt2 = [maxWidth-50,maxHeight]


#-----------------------------MAIN----------------------------------
#-------------------------------------------------------------------

#行人Control
allowPassage = True
peopleViolationIn = 0
peopleViolationOut = 0
switch = '0 : PASS \n1:STOP'

#Controller
cv2.namedWindow('config')
cv2.createTrackbar(switch,'config',0,1,nothing)

#Initializa Other Variable

averageArea = 0.000  # for calculation of min/max size for contour detected
peopleIn = 0  # number of people going up
peopleOut = 0  # number of people going up
frameCounter = 0
maskT = None
passImage = None
detectedPeople = []
detectedContours = []

#take first frame of the video
_,pFrame = cap.read()       #返回两个值 一个 Boolean 一个为视频帧
#FrameCount = 0
while(cap.isOpened()):

    #chech Passage status
    status = cv2.getTrackbarPos(switch,'config')
    #根据选择条状态来判断是否开启行人监控
    if status == 0:
        allowPassage = True
    else:
        allowPassage = False

    #Reinitialize
    frameInfo = np.ones((400,500,3),np.uint8)*255      #初始化一个三维数组元素全为0 作为画布
    averageArea = averageSize()                     #averageSize() 计算检测队列中平均面积大小
    ret,frame = cap.read()      #read a frame     ret为boolean类型 正确获取返回true
    frameForView = frame.copy()

    #FrameCount += 1
    #print(FrameCount)

    # Clean Frame
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   #转化为灰度值 易于处理

    #获取高斯混合的背景前景分割算法生成的 前景
    fgmask = fgbg.apply(gray)
    ##中值滤波进行去噪
    blur = cv2.medianBlur(fgmask,5)
    #二值化   阈值为 127      maxval为 255
    thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]  # 高斯背景法中 阴影的色值为 grey = 127

    sobelx = cv2.Sobel(thresh,cv2.CV_32F,1,0,ksize = 3)
    sobely = cv2.Sobel(thresh,cv2.CV_32F,0,1,ksize = 3)
    sobelDst = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    sobelDst = cv2.convertScaleAbs(sobelDst)

    #定义结构元素，然后进行 开闭运算  获取轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #闭运算填充内部孔洞
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #开运算去除噪点
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    Dst = cv2.bitwise_or(opening, sobelDst)

    #此处获取外部轮廓
    contours = cv2.findContours(Dst.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]

    mask_opening = cv2.inRange(Dst, np.array([0]), np.array([128]))

    #截取mask_opening区域的图像
    noBg = cv2.bitwise_and(frame, frame, mask=mask_opening)

    for c in contours:
        #通过大小过滤轮廓
        if len(humanSizeSample) < 100:
            if cv2.contourArea(c) < minArea or cv2.contourArea(c) > maxArea:
                continue
            else:
                humanSizeSample.append(cv2.contourArea(c))              #把轮廓面积值添加入humanSizeSample
        else:
            if cv2.contourArea(c) < averageArea/2 or cv2.contourArea(c) > averageArea * 3:
                continue
        (x,y,w,h) = cv2.boundingRect(c)
        detectedContours.append(Position(x,y,w,h))

    #Process Detected People
    if len(humanSizeSample) != 0:
        for people in detectedPeople:
            ##Setup Meanshift/Camshift for Tracking
            track_window = (int(people.xp()), int(people.yp()), int(people.wp()), int(people.hp()))
            hsv_roi = pOpening[int(people.yp()):int(people.yp())+int(people.hp()),int(people.xp()):int(people.xp())+int(people.wp())]

            #在两个阈值内的像素值设置为白色，不在阈值内的像素值设为黑色
            mask = cv2.inRange(hsv_roi,np.array(128),np.array(256))

            #   图像直方图
            #   cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist
            #   hist:使用多少个柱子
            #   ranges:像素值范围
            roi_hist = cv2.calcHist([hsv_roi],[0],mask,[100],[0,256])
            #归一化到范围内
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
            #指定停止条件
            #倒数第二个参数为迭代次数 原本为1，调整为10看看效果，倒数第一个参数为收敛阈值
            term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)
            #反向投影，用作Camshift追踪法参数，
            #首先为一张包含查找目标的图像创建直方图，然后把这个颜色直方图投影到输入图像寻找目标
            #也就是找到输入图像的每一个像素点的像素值在直方图中的概率，最后进行适当二值化
            #第一个参数是输入图象  第二个参数是通道  第三个参数是图象的直方图  第四个是直方图的变化范围
            dst = cv2.calcBackProject([Dst], [0], roi_hist, [0, 256], 1)
            ret,track_window = cv2.CamShift(dst, track_window, term_criteria)

            #Process POST Tracking

            #查找旋转矩形的四个顶点  返回值为list
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frameForView,[pts],True,people.colorp(),2)
            pos = sum(pts)/len(pts)             # sum 对列表内元素求和，len 可以表示有几个点  所以pos为中心点
            isFound = False

            for dC in detectedContours:
                if dC.x - toleranceRange < pos[0] < dC.x + dC.w + toleranceRange \
                        and dC.y - toleranceRange < pos[1] < dC.y + dC.h + toleranceRange:
                    people.set("x", dC.xp())
                    people.set("y", dC.yp())
                    people.set("w", dC.wp())
                    people.set("h", dC.hp())
                    people.set("speed", pos - people.centerp())
                    people.set("center", pos)
                    people.set("missingCount", 0)
                    detectedContours.remove(dC)
                    isFound = True

                    tR = getExteriorRect(pts)
                    people.set("roi", frame[tR[1]:tR[1] + tR[3], tR[0]:tR[0] + tR[2]])

                    # Process Continuous Tracking
                    prevInStatus = people.isInp()
                    currInStatus = checkPosition(boundaryPt1, boundaryPt2, people.center, inCriterion)
                    people.isIn = currInStatus

                    #Check In/Out Status Change
                    if prevInStatus != currInStatus and people.isInChangeFrameCount >= toleranceCountIOStatus:
                        if not allowPassage:
                            passImage = people.roi
                        people.set("isInChangeFrameCount",0)
                        if currInStatus:
                            peopleIn += 1
                            if not allowPassage:
                                peopleViolationIn += 1
                        else:
                            peopleOut += 1
                            if not allowPassage:
                                peopleViolationOut += 1
                    else:
                        people.set("isInChangeFrameCount",people.isInChangeFrameCount + 1)
            #Process DIS-continuous Tracking
            if not isFound:
                if people.missingCount > toleranceCount:
                    detectedPeople.remove(people)
                else:
                    if checkTouchSide(people.x + people.speed[0], people.y + people.speed[1], people.w,
                                       people.h, maxWidth, maxHeight, toleranceRange):
                        detectedPeople.remove(people)
                    else:
                        people.set("missingCount", people.missingCount + 1)
                        people.set("x", people.x + people.speed[0])
                        people.set("y", people.y + people.speed[1])
                        people.set("center", people.center + people.speed)

    #check New people
    for dC in detectedContours:
        if checkTouchSide(dC.x,dC.y, dC.w, dC.h,maxWidth,maxHeight,toleranceRange):
            startHue += hueIncrementValue
            detectedPeople.append(People(dC.x, dC.y, dC.w, dC.h, frame[dC.y:dC.y+dC.h, dC.x:dC.x+dC.w], startHue))

    #Re-set
    detectedContours = []
    pFrame = frame
    pNoBg = noBg
    pOpening = Dst
    frameCounter += 1

    #output
    try:
        # Setup Stats
        textNoOfPeople = "People: " + str(len(detectedPeople))
        textNoIn = "In: " + str(peopleIn)
        textNoOut = "Out: " + str(peopleOut)
        textNoViolationIn = "In: " + str(peopleViolationIn)
        textNoViolationOut = "Out: " + str(peopleViolationOut)

        if allowPassage:
            cv2.line(frameForView, (int(boundaryPt1[0]), int(boundaryPt1[1])),
                     (int(boundaryPt2[0]), int(boundaryPt2[1])), (0, 255, 0), 1)
        else:
            cv2.line(frameForView, (int(boundaryPt1[0]), int(boundaryPt1[1])),
                     (int(boundaryPt2[0]), int(boundaryPt2[1])), (0, 0, 255), 1)

        # Draw Infos
        cv2.putText(frameInfo, textNoOfPeople, (30, 40), cv2.FONT_HERSHEY_SIMPLEX
                    , 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frameInfo, textNoIn, (30, 80), cv2.FONT_HERSHEY_SIMPLEX
                    , 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frameInfo, textNoOut, (30, 120), cv2.FONT_HERSHEY_SIMPLEX
                    , 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(frameInfo, (0, 160), (640, 160), (255, 255, 255), 1)
        cv2.putText(frameInfo, "VIOLATION", (30, 200), cv2.FONT_HERSHEY_SIMPLEX
                    , 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frameInfo, textNoViolationIn, (30, 240), cv2.FONT_HERSHEY_SIMPLEX
                    , 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frameInfo, textNoViolationOut, (30, 280), cv2.FONT_HERSHEY_SIMPLEX
                    , 1, (0, 0, 0), 1, cv2.LINE_AA)

        # Display
        cv2.imshow('FrameForView', frameForView)
        ##        cv2.imshow('Frame', frame)
        if passImage != None:
            cv2.imshow('Violators', passImage)
        cv2.imshow('config', frameInfo)

    except Exception as e:
        print(e)
        break

        # Abort and exit with ESC
    k = cv2.waitKey(30) & 0xff
    if k == ord(' '):
        cv2.waitKey(0)
    if k == 27:
        break
    # else:
##    cv2.imwrite(chr(k) + ".jpg", frame)

cap.release()
cv2.destroyAllWindows()