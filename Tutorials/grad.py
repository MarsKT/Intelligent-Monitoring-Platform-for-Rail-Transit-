import cv2 as cv
def sobel_demo(image):
    grad_x = cv.Sobel(image,cv.CV_32F,1,0)
    grad_y = cv.Sobel(image,cv.CV_32F,0,1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    #cv.imshow("grad_x",gradx)
    #cv.imshow("grad_y",grady)
    gradexy = cv.addWeighted(gradx,0.5,grady,0.5,0)
    cv.namedWindow("sobel", cv.WINDOW_NORMAL)
    cv.imshow("sobel",gradexy)
def scharr_demo(image):
    grad_x = cv.Scharr(image,cv.CV_32F,1,0)
    grad_y = cv.Scharr(image,cv.CV_32F,0,1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    #cv.imshow("grad_x", gradx)
    #cv.imshow("grad_y", grady)
    gradexy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.namedWindow("scharr", cv.WINDOW_NORMAL)
    cv.imshow("scharr", gradexy)
def laplace_demo(image):
    dst = cv.Laplacian(image,cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv.namedWindow("Laplace", cv.WINDOW_NORMAL)
    cv.imshow("Laplace",lpls)
filepath = "D:/PycharmProject/Image/face3.jpg"
src = cv.imread(filepath)
cv.namedWindow("input_image",cv.WINDOW_NORMAL)
cv.imshow("input_image",src)
sobel_demo(src)
scharr_demo(src)
laplace_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()