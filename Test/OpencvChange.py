import cv2

filepath = "D:/PycharmProject/Image/wolf.jpg"
img = cv2.imread(filepath)
#转换灰色
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

x = y = 10
w = 100
color = (255,255,255)
cv2.rectangle(gray,(x,y),(x+w,y+w),color,1)#绘制矩形
cv2.imshow("Image",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()