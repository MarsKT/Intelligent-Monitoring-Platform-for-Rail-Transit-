import cv2
import numpy as np

filepath = "D:/PycharmProject/Image/face2.jpg"
img = cv2.imread(filepath)
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], np.float32)
dst1 = cv2.filter2D(img,-1,kernel)
cv2.imshow("image1",dst1)
cv2.imshow("image2",img)
c = cv2.waitKey(10)
cv2.waitKey(0)
cv2.destroyAllWindows()
