import cv2 as cv
def edge_demo(image):
    blurred = cv.GaussianBlur(image,(3,3),0)
    gray = cv.cvtColor(blurred,cv.COLOR_RGB2GRAY)
    edge_output = cv.Canny(gray,50,150)
    cv.imshow("Canny Edge",edge_output)
    dst = cv.bitwise_and(image,image,mask = edge_output)
    cv.imshow("Color Edge",dst)
filepath = "D:/PycharmProject/Image/face2.jpg"
src = cv.imread(filepath)
cv.namedWindow("input_image",cv.WINDOW_NORMAL)
cv.imshow("input_image",src)
edge_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()