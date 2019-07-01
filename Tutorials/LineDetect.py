import numpy as np
import cv2 as cv
#标准霍夫线变换
def line_detection(image):   #此函数画出图形有问题，之后改正
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize=3)
    #cv.imshow("edges",edges)
    lines = cv.HoughLines(edges,1,np.pi/180,80)
    for line in lines:
        rho,theta = line[0]    #rho为极径 theta为极角用弧度表示
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho           #代表 x = r * cos(theta)
        y0 = b * rho
        x1 = int(x0 + 100 * (-b))  # 计算直线起点横坐标
        y1 = int(y0 + 100 * a)  # 计算起始起点纵坐标
        x2 = int(x0 - 100 * (-b))  # 计算直线终点横坐标
        y2 = int(y0 - 100 * a)  # 计算直线终点纵坐标
        cv.line(image,(x1,y1),(x2,y2),(0,0,255),1)   # 注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        cv.imshow("image_line",image)
#统计概率霍夫变换
def line_detect_possible(image):
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=5)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("line_detect_possible_demo", image)
filepath = "D:/PycharmProject/Image/street.jpg"
src = cv.imread(filepath)
print(src.shape)
cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
cv.imshow('input_image', src)
line_detection(src)
src = cv.imread(filepath)
line_detect_possible(src)
cv.waitKey(0)
cv.destroyAllWindows()