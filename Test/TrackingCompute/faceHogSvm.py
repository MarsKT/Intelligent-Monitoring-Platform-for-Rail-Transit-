import cv2
import numpy as np
import random
def load_negimages(dirname,amout = 9999):
    img_list = []
    file = open(dirname)
    img_name = file.readline()
    while img_name != '':  #文件尾
        img_name = dirname.rsplit(r'/',1)[0] + r'/' + img_name.split('/',1)[1].strip('\n')
        print(img_name)
        img = cv2.imread(img_name)
        #print(img)
        img = cv2.resize(img,(64,128))
        #print(img)
        img_list.append(img)
        img_name = file.readline()
        amout -= 1
        if amout <= 0:
            break
    return img_list
def load_posimages(dirname,amout = 9999):
    img_list = []
    file = open(dirname)
    img_name = file.readline()
    while img_name != '':  #文件尾
        img_name = dirname.rsplit(r'/',1)[0] + r'/' + img_name.strip('\n')
        print(img_name)
        img = cv2.imread(img_name)

        img = cv2.resize(img,(64,128))
        #print(img)
        img_list.append(img)
        img_name = file.readline()
        amout -= 1
        if amout <= 0:
            break
    return img_list


#从每一张没有人脸的原始图片中随即裁剪出10张64 128 的图片作为负样本
def sample_neg(full_neg_list, neg_list, size):
    random.seed(1)
    width,height = size[0],size[1]
    for i in range(len(full_neg_list)):
        for j in range(5):
            y = int(random.random() * (len(full_neg_list[i])-height))
            x = int(random.random() * (len(full_neg_list[i][0])-width))
            neg_list.append(full_neg_list[i][y:y + height,x:x+width])
    return neg_list

def computeHogs(img_list,gradient_lst,wsize=(64,128)):
    hog = cv2.HOGDescriptor()
    count = 0
    for i in range(len(img_list)):
        gray = cv2.cvtColor(img_list[i],cv2.COLOR_BGR2GRAY)
        #print(i)
        gradient_lst.append(hog.compute(gray))
        count += 1
    return count

def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    rho,_,_ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv,[[-rho]],0)

#第一步 计算HOG特征
neg_list = []
pos_list = []
gradient_lst = []
labels = []
hard_neg_list = []

pos_list = load_posimages('D:/GraduationProject/Project1/Image/pos/pos.txt')
full_neg_lst = load_negimages('D:/GraduationProject/Project1/Image/INRIAPerson/INRIAPerson/Train/neg.lst')
sample_neg(full_neg_lst,neg_list,[64,128])
print('pos num = ',len(pos_list))
print('neg num = ',len(neg_list))
count = computeHogs(pos_list,gradient_lst)
print('pos HogDes = %d'%count)
print('before')
[labels.append(+1) for _ in range(len(pos_list))]

count = computeHogs(neg_list, gradient_lst)
print('neg HogDes = %d'%count)
[labels.append(-1) for _ in range(len(neg_list))]
print('after')
#创建SVM
svm = cv2.ml.SVM_create()
#默认为C_SVC 分类器
#训练svm
#OpenCV中SVM参数有9种
#kernaltype 核函数类型   svmtype Svm类型   degree 内核函数参数   gamma 内核函数参数    coef0 内核函数参数
#Cvalue svm参数类型     nu p 都为svm参数    classWeight C_svc权重，与惩罚系数有关     termcrit 中止条件
# svm.setCoef0(0)
# svm.setCoef0(0.0)
# svm.setDegree(3)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
svm.setTermCriteria(criteria)
#svm.setGamma(0)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setNu(0.5)
svm.setP(0.1)
svm.setC(0.01)  # 允许存在的误差系数
svm.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))

print('svm complate')
#第三步加入识别错误样本进行第二轮训练

hog = cv2.HOGDescriptor()
hard_neg_list.clear()
hog.setSVMDetector(get_svm_detector(svm))
for i in range(len(full_neg_lst)):
    rects, wei = hog.detectMultiScale(full_neg_lst[i], winStride=(4, 4),padding=(8, 8), scale=1.05)
    for (x,y,w,h) in rects:
        hardExample = full_neg_lst[i][y:y+h, x:x+w]
        hard_neg_list.append(cv2.resize(hardExample,(64,128)))
computeHogs(hard_neg_list, gradient_lst)
[labels.append(-1) for _ in range(len(hard_neg_list))]
svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))

hog.setSVMDetector(get_svm_detector(svm))
hog.save('myHogDector.bin')
