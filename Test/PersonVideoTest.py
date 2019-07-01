import cv2 as cv
import numpy as np
def is_inside(o,i):
    ox,oy,ow,oh = o
    ix,iy,iw,ih = i
    return ox>ix and oy>ox and ox+ow < ix+iw and oy+oh < iy+ih
def draw_person(img,person):
    x,y,w,h = person
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)

filepath = "D:\GraduationProject\Project1\Image/PersonTest3.mp4"
cap = cv.VideoCapture(filepath)
#ret,frame = cap.read()
#rows,cols = frame.shape[:2]
#scale = 1.0;

#创建HOG描述符对象
hog = cv.HOGDescriptor()
detector = cv.HOGDescriptor_getDefaultPeopleDetector()
print('detector', type(detector), detector.shape)
hog.setSVMDetector(detector)
while(1):
    ret,frame = cap.read()
    if ret == True:
        found,w = hog.detectMultiScale(frame)
        #print('found', type(found), found.shape)
        found_filtered = []
        for ri,r in enumerate(found):
            for qi,q in enumerate(found):
                if ri != qi and is_inside(r,q):
                    break;
                else:
                    found_filtered.append(r)
        for person in found_filtered:
            draw_person(frame,person)
        cv.imshow("frame",frame)
        k = cv.waitKey(60) & 0xff
        if k == 27:
            break;
    else:
        break;
cv.destroyAllWindows()
cap.release()



