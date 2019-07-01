import numpy as np
import cv2 as cv
def Template_Demo():
    filepath = "D:/PycharmProject/Image/face1.jpg"
    target = cv.imread(filepath)
    tpl = target[100:200,100:200]
    cv.namedWindow("template image",cv.WINDOW_NORMAL)
    cv.imshow("template image",tpl)
    cv.namedWindow("target image",cv.WINDOW_NORMAL)
    cv.imshow("taret image",target)
    method = [cv.TM_SQDIFF_NORMED,cv.TM_CCORR_NORMED,cv.TM_CCOEFF_NORMED]
    th,tw= tpl.shape[:2]
    for md in method:
        print(md)
        result = cv.matchTemplate(target,tpl,md)
        min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw,tl[1]+th)
        cv.rectangle(target,tl,br,(0,0,255),2)
        cv.namedWindow("match-"+ np.str(md),cv.WINDOW_NORMAL)
        cv.imshow("match-"+np.str(md),target)
Template_Demo()
cv.waitKey(0)
cv.destroyAllWindows()
