from __future__ import print_function
import cv2 as cv
import numpy as np
import sys

filepath = "D:/PycharmProject/Image/face1.jpg"
path = "D:/PycharmProject/Image/face10.jpg"
fac = cv.imread(path)
img = cv.imread(filepath,cv.IMREAD_GRAYSCALE)
rows,cols = img.shape
m = cv.getOptimalDFTSize(rows)
n = cv.getOptimalDFTSize(cols)
padded = cv.copyMakeBorder(img,0,m-rows,0,n-cols,cv.BORDER_CONSTANT,value=[0,0,0])
# print(padded.shape)
# padded.dtype = np.float32
# print(padded.shape)
#print(padded.dtype)
planes = [np.float32(padded),np.zeros(padded.shape,np.float32)]
complexI = cv.merge(planes)
cv.dft(complexI,complexI)
cv.split(complexI,planes)
# print(planes[0])
cv.magnitude(planes[0],planes[1],planes[0]) #magnitude()的意义为根号下平方和 即 计算辐值
magI = planes[0]

matOfones = np.ones(magI.shape,dtype=magI.dtype)
cv.add(matOfones,magI,magI)
cv.log(magI,magI)
#print(magI.dtype)
magI_rows,magI_cols = magI.shape
magI = magI[0:(magI_rows & -2),0:(magI_cols & -2)]
cx = int(magI_rows/2)
cy = int(magI_cols/2)

q0 = magI[0:cx, 0:cy]  # Top-Left - Create a ROI per quadrant
q1 = magI[cx:cx + cx, 0:cy]  # Top-Right
q2 = magI[0:cx, cy:cy + cy]  # Bottom-Left
q3 = magI[cx:cx + cx, cy:cy + cy] # Bottom-Right

tmp = np.copy(q0)  # swap quadrants (Top-Left with Bottom-Right)
magI[0:cx, 0:cy] = q3
magI[cx:cx + cx, cy:cy + cy] = tmp
tmp = np.copy(q1)  # swap quadrant (Top-Right with Bottom-Left)
magI[cx:cx + cx, 0:cy] = q2
magI[0:cx, cy:cy + cy] = tmp
print(magI)
cv.normalize(magI,magI,0,1,cv.NORM_MINMAX)
print(magI.dtype)
cv.imshow("inputimage",img)
cv.imshow("spectrum magnitude",magI)
cv.waitKey(0)
