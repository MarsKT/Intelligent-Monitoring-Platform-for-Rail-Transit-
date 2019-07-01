import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import tkinter as tk
from tkinter import filedialog
pi180 = 180 / math.pi

data = []
pArray = []
VertexX = []
VertexY = []

class Vertex:

    def __init__(self,x,y):
        self.x = x
        self.y = y

        self.connection = []
        self.angle = 0
        self.flag = True
        self.preVertex = None
        self.nextVertex = None
        self.color = None

#存储三角形顶点列表
Trangle = []
def getAngle(p1x,p1y,p2x,p2y,p3x,p3y):
    cos1 = getCos(p1x,p1y,p2x,p2y,p3x,p3y)
    return math.acos(cos1) * pi180

def getCos(p1x,p1y,p2x,p2y,p3x,p3y):
    length1_2 = getLength(p1x,p1y,p2x,p2y)
    length1_3 = getLength(p1x, p1y, p3x, p3y)
    length2_3 = getLength(p2x, p2y, p3x, p3y)
    res = (math.pow(length1_2,2) + math.pow(length1_3,2) - math.pow(length2_3,2))/(2*length1_2*length1_3)
    return res

def getLength(p1x,p1y,p2x,p2y):
    diff_x = math.fabs(p2x-p1x)
    diff_y = math.fabs(p2y-p1y)
    length_pow = math.pow(diff_x,2) + math.pow(diff_y,2)
    return math.sqrt(length_pow)

def isIn(n,vx,vy,x,y):
    i = 0
    j = n-1
    c = False
    for i in range(n):
        if (( (vy[i]>y) != (vy[j]> y) ) and (x < (vx[j]-vx[i])*(y-vy[i]) / (vy[j]-vy[i]) + vx[i]) ):
            c = not c
        j = i
    return c

#坐标法求三角形面积
def IsTrangleOrArea(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def IsInTrangle(x1, y1, x2, y2, x3, y3, x, y):
    if (x1 == x and y1 == y) or (x2 == x and y2 == y) or (x3 == x and y3 == y):
        return False
    # 三角形ABC的面积
    ABC = IsTrangleOrArea(x1, y1, x2, y2, x3, y3)
    # 三角形PBC的面积
    PBC = IsTrangleOrArea(x, y, x2, y2, x3, y3)
    # 三角形ABC的面积
    PAC = IsTrangleOrArea(x1, y1, x, y, x3, y3)
    # 三角形ABC的面积
    PAB = IsTrangleOrArea(x1, y1, x2, y2, x, y)
    return (ABC == PBC + PAC + PAB)


#判断邻接点里是否含有某种颜色，如果包含返回false
def isOk(v,c):
    res = True
    for i in v.connection:
        if i.color == c:
            res = False
            break
    return res

onePath = []
allPath = []
def dfs(depth,n):
    if depth > n:
        allPath.append(onePath.copy())
    else:
        for i in range(3):
            if isOk(pArray[depth],i):
                onePath.append(i)
                pArray[depth].color = i
                dfs(depth+1,n)
                # if len(onePath) != 0:
                #     onePath.pop()
                onePath.pop()
                pArray[depth].color = None

def start():
    # 坐标数据文本
    file = open(path.get(), 'r')
    context = file.readlines()
    data.clear()
    pArray.clear()
    VertexY.clear()
    VertexX.clear()
    # data = []
    # pArray = []
    # VertexX = []
    # VertexY = []

    # 去掉末尾回车键
    for i in range(len(context)):
        if i != len(context) - 1:
            data.append(context[i][:-1].split(' '))
        else:
            data.append(context[i].split(' '))

    # 提取坐标点
    for i in range(len(data)):
        VertexX.append(float(data[i][0]))
        VertexY.append(float(data[i][1]))
        cur = Vertex(float(data[i][0]), float(data[i][1]))
        if i != 0:
            # print(pArray[i-1].x,end=' ')
            # print(pArray[i-1].y)
            cur.preVertex = pArray[i - 1]
            pArray[i - 1].nextVertex = cur
            # 添加联通分量
            # cur.connection.append(pArray[i-1])
            # pArray[i - 1].connection.append(cur)
            if i == len(data) - 1:
                # print(pArray[i - 1].x, end=' ')
                # print(pArray[i - 1].y)
                cur.nextVertex = pArray[0]
                pArray[0].preVertex = cur
                # 添加联通分量
                # cur.connection.append(pArray[0])
                # pArray[0].connection.append(cur)
        pArray.append(cur)

    for i in pArray:
        i.connection.append(i.preVertex)
        i.connection.append(i.nextVertex)

    # 计算每个顶点角度
    for i in pArray:
        GP = [(i.x + i.preVertex.x + i.nextVertex.x) / 3, (i.y + i.preVertex.y + i.nextVertex.y) / 3]
        if isIn(len(data), VertexX, VertexY, GP[0], GP[1]):
            i.angle = getAngle(i.x, i.y, i.preVertex.x, i.preVertex.y, i.nextVertex.x, i.nextVertex.y)
        else:
            i.angle = 360 - getAngle(i.x, i.y, i.preVertex.x, i.preVertex.y, i.nextVertex.x, i.nextVertex.y)
        # print(i.angle)

    CopypArry = pArray.copy()

    # 打印坐标和角度 用于检测
    # for i in CopypArry:
    #     print(i.x,end=' ')
    #     print(i.angle)

    CopypArrySave = []
    # 手动测试一下数据正确性
    # print(pArray[3].preVertex.x+' '+pArray[3].nextVertex.x)
    curIndex = 0
    while (True):
        if len(CopypArry) == 3:
            curA = CopypArry[0]
            curB = CopypArry[1]
            curC = CopypArry[2]
            CopypArrySave.append(curA)
            CopypArrySave.append(curB)
            CopypArrySave.append(curC)
            if curB not in curA.connection:
                curA.connection.append(curB)
            if curC not in curA.connection:
                curA.connection.append(curC)

            if curA not in curB.connection:
                curB.connection.append(curA)
            if curC not in curB.connection:
                curB.connection.append(curC)

            if curB not in curC.connection:
                curC.connection.append(curB)
            if curA not in curC.connection:
                curC.connection.append(curA)

            Trangle.append([[CopypArry[0].x, CopypArry[0].y], [CopypArry[1].x, CopypArry[1].y],
                            [CopypArry[2].x, CopypArry[2].y]])
            break
        if CopypArry[curIndex].angle >= 180 or CopypArry[curIndex].flag == False:
            curIndex = (1 + curIndex)  # %len(CopypArry)
            # print(curIndex)
            continue
        else:
            curFlag = False
            for i in CopypArry:
                if IsInTrangle(CopypArry[curIndex].x, CopypArry[curIndex].y, CopypArry[curIndex].preVertex.x
                        , CopypArry[curIndex].preVertex.y, CopypArry[curIndex].nextVertex.x,
                               CopypArry[curIndex].nextVertex.y, i.x, i.y):
                    curFlag = True
                    break
            if curFlag == True:
                CopypArry[curIndex].flag = False
                curIndex = (1 + curIndex)  # % len(CopypArry)
                # print(curIndex)
                continue
            else:
                Trangle.append([[CopypArry[curIndex].x, CopypArry[curIndex].y],
                                [CopypArry[curIndex].preVertex.x, CopypArry[curIndex].preVertex.y],
                                [CopypArry[curIndex].nextVertex.x, CopypArry[curIndex].nextVertex.y]])
                CopypArry[curIndex].nextVertex.preVertex = CopypArry[curIndex].preVertex
                CopypArry[curIndex].preVertex.nextVertex = CopypArry[curIndex].nextVertex
                pre = CopypArry[curIndex].preVertex
                next = CopypArry[curIndex].nextVertex
                del CopypArry[curIndex]
                # 添加联通分量
                pre.connection.append(next)
                next.connection.append(pre)

                # 重新计算O Q 的angle，并将flag置为True
                pre.flag = True
                next.flag = True
                CopyVertexX = []
                CopyVertexY = []
                for i in CopypArry:
                    CopyVertexX.append(i.x)
                    CopyVertexY.append(i.y)
                # 重新计算角度
                curGP = [(pre.x + pre.preVertex.x + pre.nextVertex.x) / 3,
                         (pre.y + pre.preVertex.y + pre.nextVertex.y) / 3]
                if isIn(len(CopypArry), CopyVertexX, CopyVertexY, curGP[0], curGP[1]):
                    pre.angle = getAngle(pre.x, pre.y, pre.preVertex.x, pre.preVertex.y, pre.nextVertex.x,
                                         pre.nextVertex.y)
                else:
                    pre.angle = 360 - getAngle(pre.x, pre.y, pre.preVertex.x, pre.preVertex.y, pre.nextVertex.x,
                                               pre.nextVertex.y)
                # print('preAngle = %f' %pre.angle)

                curGP = [(next.x + next.preVertex.x + next.nextVertex.x) / 3,
                         (next.y + next.preVertex.y + next.nextVertex.y) / 3]
                if isIn(len(CopypArry), CopyVertexX, CopyVertexY, curGP[0], curGP[1]):
                    next.angle = getAngle(next.x, next.y, next.preVertex.x, next.preVertex.y, next.nextVertex.x,
                                          next.nextVertex.y)
                else:
                    next.angle = 360 - getAngle(next.x, next.y, next.preVertex.x, next.preVertex.y, next.nextVertex.x,
                                                next.nextVertex.y)
                # print('nextAngle = %f' % next.angle)

                CopypArrySave.append(CopypArry[curIndex])
    # 进行染色
    dfs(0, len(pArray) - 1)
    num0 = allPath[0].count(0)
    num1 = allPath[0].count(1)
    num2 = allPath[0].count(2)
    minNum = min(num0, num1, num2)  #染色数最少的颜色

    maxX = int(max(VertexX))
    maxY = int(max(VertexY))

    print(maxX)
    print(maxY)
    ###########################################################################
    a = np.array([data], dtype=np.int32)
    im1 = np.ones([maxY + 20, maxX + 20, 3], dtype=np.int32) * 255
    #im1.fill(255)
    # im2 = np.zeros([175,175],dtype=np.int32)
    if minNum == num0:
        for i in range(len(allPath[0])):
            if allPath[0][i] == 0:
                cv2.circle(im1, (int(pArray[i].x), int(pArray[i].y)), 4, (0, 255, 0), -1)
    elif minNum == num1:
        for i in range(len(allPath[0])):
            if allPath[0][i] == 1:
                cv2.circle(im1, (int(pArray[i].x), int(pArray[i].y)), 4, (0, 255, 0), -1)
    elif minNum == num2:
        for i in range(len(allPath[0])):
            if allPath[0][i] == 2:
                cv2.circle(im1, (int(pArray[i].x), int(pArray[i].y)), 4, (0, 255, 0), -1)
    #cv2.polylines(im1, a, True, (0, 0, 0))
    # for i in range(len(data)):
    #     if allPath[0][i] == 0:
    #         cv2.circle(im1, (int(data[i][0]), int(data[i][1])), 4, (255, 0, 0), -1)
    #     elif allPath[0][i] == 1:
    #         cv2.circle(im1, (int(data[i][0]), int(data[i][1])), 4, (0, 255, 0), -1)
    #     elif allPath[0][i] == 2:
    #         cv2.circle(im1, (int(data[i][0]), int(data[i][1])), 4, (0, 0, 255), -1)
    for i in Trangle:
        b = np.array([i], dtype=np.int32)
        cv2.polylines(im1, b, 1, (0,0,0))
    # #绘制连线
    # for i in Trangle:
    #     cv2.line(im,(int(i[1][0]),int(i[1][1])),(int(i[2][0]),int(i[2][0])),255)
    plt.suptitle('StationFrame', fontsize=20)
    plt.imshow(im1)
    # plt.imshow(im2)
    plt.show()
    #plt.close()
def selectPath():
    selectpath = filedialog.askopenfilename()
    path.set(selectpath)
root = tk.Tk()
root.title('辅助分析')
path = tk.StringVar()
lable = tk.Label(root,text='坐标文件位置: ').grid(row= 1, column = 1)
entry = tk.Entry(root,textvariable = path).grid(row = 1,column = 2)
button = tk.Button(root,text='路径选择',command = selectPath).grid(row = 1, column = 3)
start = tk.Button(root,text='推荐部署点',command = start).grid(row = 2,column = 2)

root.mainloop()
