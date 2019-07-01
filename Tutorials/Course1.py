import cv2
import numpy as np
#start = cv2.getTickCount()
#测试代码
#z = np.uint8
#end = cv2.getTickCount()
#print((end - start)/cv2.getTickFrequency())
a = {'a','b','c'}
# #a[5] = 'nihao'
# x = np.array([[1,2],[3,4]], dtype=np.float64)
# y = np.array([[5,6],[7,8]], dtype=np.float64)
#
# z = np.tile(x,(3,2))
# print(z)
# print(x.T)
#
# m = np.empty((2,3))
# print(m)

# n = np.array([[1,-1,2]])
# #m = np.transpose(n)
# m = n.T
# print(m)
A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
b = np.transpose(np.array([[-3,5,-2]]))
x = np.linalg.solve(A,b)
print(x)