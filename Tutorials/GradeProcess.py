import cv2
import numpy as np
filepath = "D:\GraduationProject\Project1\Image/1.jpg"
img = cv2.imread(filepath)
filp = cv2.flip(img,1)

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
#kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
dst = cv2.filter2D(img, -1, kernel=kernel)
cv2.imwrite("D:\GraduationProject\Project1\Image/1test.jpg",dst)
# cv2.imshow("img",dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()