import numpy as np
import math
import cv2
import dlib
import tkinter as tk
#--------------------------------------------------------------------------------
# #测试isIn函数
# class Vertex:
#     angle = 0
#     flag = True
#     preVertex = None
#     nextVertex = None
#     def __init__(self,x,y):
#         self.x = x
#         self.y = y
# def isIn(n,vx,vy,x,y):
#     i = 0
#     j = n-1
#     c = False
#     for i in range(n):
#         if (( (vy[i]>y) != (vy[j]> y) ) and (x < (vx[j]-vx[i])*(y-vy[i]) / (vy[j]-vy[i]) + vx[i]) ):
#             c = not c
#         j = i
#     return c
# file = open('TrackingCompute/point.txt','r')
# context = file.readlines()
# data = []
# pArray = []
# VertexX = []
# VertexY = []
# #去掉末尾回车键
# for i in range(len(context)):
#     if i != len(context)-1:
#         data.append(context[i][:-1].split(' '))
#     else:
#         data.append(context[i].split(' '))
# #提取坐标点
# for i in range(len(data)):
#     VertexX.append(float(data[i][0]))
#     VertexY.append(float(data[i][1]))
#     cur = Vertex(float(data[i][0]),float(data[i][1]))
#     if i != 0:
#         cur.preVertex = pArray[i-1]
#         pArray[i-1].nextVertex = cur
#         if i == len(data)-1:
#             cur.nextVertex = pArray[0]
#             pArray[0].preVertex = cur
#     pArray.append(cur)
# print(isIn(len(data),VertexX,VertexY,10,90))
#------------------------------------------------------------------------------------------------
#
# filepath = 'D:\GraduationProject\Project1\Image/passageway1.avi'
# cap = cv2.VideoCapture(filepath)
# Framecount = 0
# while cap.isOpened():
#     ret,frame = cap.read()
#     Framecount += 1
#     print(Framecount)
#     cv2.imshow('frame',frame)
#     k = cv2.waitKey(30) & 0xff
#     if k == ord(' '):
#         cv2.waitKey(0)
#     if k == 27:
#         break
# cv2.destroyAllWindows()
# cap.release()
root = tk.Tk()
text1 = tk.StringVar()
text2 = tk.StringVar()
text3 = tk.StringVar()
text4 = tk.StringVar()
text5 = tk.StringVar()
text6 = tk.StringVar()
text7 = tk.StringVar()
text8 = tk.StringVar()
text9 = tk.StringVar()
text10 = tk.StringVar()
label1 = tk.Label(root,text='0-2岁：').grid(row=0, column=0)
label2 = tk.Label(root,text='2-4岁：').grid(row=1, column=0)
label3 = tk.Label(root,text='8-13岁：').grid(row=2, column=0)
label4 = tk.Label(root,text='15-20岁：').grid(row=3, column=0)
label5 = tk.Label(root,text='25-32岁：').grid(row=4, column=0)
label6 = tk.Label(root,text='38-43岁：').grid(row=5, column=0)
label7 = tk.Label(root,text='48-53岁：').grid(row=6, column=0)
label8 = tk.Label(root,text='60+岁：').grid(row=7, column=0)
label9 = tk.Label(root,text='男：').grid(row=8, column=0)
label10 = tk.Label(root,text='女：').grid(row=9, column=0)

label11 = tk.Label(root,textvariable=text1).grid(row=0, column=1)
label22 = tk.Label(root,textvariable=text2).grid(row=1, column=1)
label33 = tk.Label(root,textvariable=text3).grid(row=2, column=1)
label44 = tk.Label(root,textvariable=text4).grid(row=3, column=1)
label55 = tk.Label(root,textvariable=text5).grid(row=4, column=1)
label66 = tk.Label(root,textvariable=text6).grid(row=5, column=1)
label77 = tk.Label(root,textvariable=text7).grid(row=6, column=1)
label88 = tk.Label(root,textvariable=text8).grid(row=7, column=1)
label99 = tk.Label(root,textvariable=text9).grid(row=8, column=1)
label1010 = tk.Label(root,textvariable=text10).grid(row=9, column=1)
text1.set(1)
all = []
all = all + [1,2,3]
print(all)
root.mainloop()