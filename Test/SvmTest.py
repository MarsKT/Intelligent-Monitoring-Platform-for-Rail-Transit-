import cv2
import numpy as np

#data
#women
rand1 = np.array([[155,48],[159,50],[164,53],[168,56],[172,60]])
#man
rand2 = np.array([[152,53],[156,55],[160,56],[172,64],[176,65]])
#tag
label = np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])

data = np.vstack((rand1,rand2))
data = np.array(data,dtype='float32')

#exercise
svm = cv2.ml.SVM_create()

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)

result = svm.train(data,cv2.ml.ROW_SAMPLE,label)

pt_data = np.vstack([[167,55],[162,57]])
pt_data = np.array(pt_data,dtype='float32')
print(pt_data)
(par1,par2) = svm.predict(pt_data)
print(par1)
print(par2)
