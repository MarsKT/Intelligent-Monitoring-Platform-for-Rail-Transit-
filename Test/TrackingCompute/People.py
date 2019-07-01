import colorsys
import cv2
import collections
import numpy as np
class People(object):
    def __init__(self,_x,_y,_w,_h,_roi,_hue):
        self.x = _x
        self.y = _y
        self.w = _w
        self.h = _h
        self.roi = _roi

        #Display of the contour while tracking
        self.hue = _hue
        #self.color = hsv2rgb(self.hue%1,1,1);

        #Motion Descriptors
        self.center = [_x+ _w/2,_y+ _h/2]
        #self.isIn =
        #self.isInChangeFrameCount =
        #self.speed = [0,0]
        #self.missingCount = 0

        #Roi - Region of Interest

        self.maxRoi = _roi
        self.roi = _roi
    def x(self):
        return self.x

    def y(self):
        return self.y

    def w(self):
        return self.w

    def h(self):
        return self.h

    def roi(self):
        return self.roi

    def color(self):
        return self.color

    def center(self):
        return self.center

    def maxRoi(self):
        return self.maxRoi

    def isIn(self):
        return self.isIn

    def speed(self):
        return self.speed

    def missingCount(self):
        return self.missingCount

    def isInChangeFrameCount(self):
        return self.isInChangeFrameCount

    def set(self,name,value):
        if name == 'x':
            self.x = value
        elif name == 'y':
            self.y = value
        elif name == 'w':
            self.w = value
        elif name == 'h':
            self.h = value
        elif name == 'center':
            self.center = value
        elif name == 'roi':
            self.roi = value
            if self.roi.shape[0]*self.roi.shape[1] > self.maxRoi.shape[0]*self.maxRoi.shape[1]:
                self.maxRoi = self.roi
        elif name == "speed":
            self.speed = value
        elif name == "missingCount":
            self.missingCount = value
        elif name == "isIn":
            self.isIn = value
        elif name == "isInChangeFrameCount":
            self.isInChangeFrameCount = value
        else:
            return
