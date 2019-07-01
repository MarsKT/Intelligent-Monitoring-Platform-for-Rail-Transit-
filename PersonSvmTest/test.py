import numpy as np
import cv2 as cv
from .nums import py_cpu_nms
#from imutils.object_detection import non_max_suppression

hog = cv.HOGDescriptor()
hog.load('myHogDector1.bin')

img = cv.imread('person_236.png')

cv.imshow('src', img)
cv.waitKey(10)

rects, scores = hog.detectMultiScale(img, winStride=(8, 8), padding=(0, 0), scale=1.05)

sc = [score[0] for score in scores]
sc = np.array(sc)

# 转换下输出格式(x,y,w,h) -> (x1,y1,x2,y2)
for i in range(len(rects)):
    r = rects[i]
    rects[i][2] = r[0] + r[2]
    rects[i][3] = r[1] + r[3]

pick = []
# 非极大值移植
print('rects_len', len(rects))
pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
print('pick_len = ', len(pick))

# 画出矩形框
for (x, y, xx, yy) in pick:
    cv.rectangle(img, (x, y), (xx, yy), (0, 0, 255), 2)

cv.imshow('a', img)
cv.waitKey(10)