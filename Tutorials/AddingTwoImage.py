import cv2
import numpy as np
alpha = 0.5
input_alpha = float(input())
if 0<= alpha <=1:
    alpha = input_alpha

filepath1 = "D:/PycharmProject/Image/face1.jpg"
filepath2 = "D:/PycharmProject/Image/face2.jpg"
src1 = cv2.imread(filepath1)
src2 = cv2.imread(filepath2)

rows,cols = src2.shape[:2]
src1_dst = cv2.resize(src1,(cols,rows),interpolation=cv2.INTER_CUBIC)
beta = (1.0 - alpha)
dst = cv2.addWeighted(src1_dst,alpha,src2,beta,0.0)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()