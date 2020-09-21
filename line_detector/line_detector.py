import cv2
from pylsd.lsd import lsd
import numpy as np

img = cv2.imread('./../sample_image.jpg')
W=200
w_ratio=200/img.shape[1]
img = cv2.resize(img,(200, int(w_ratio*img.shape[0])))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray,(3,3),5)
gray2 = cv2.GaussianBlur(gray,(5,5),5)
gray3 = cv2.GaussianBlur(gray,(15,15),3)
#gray1 = cv2.bitwise_not(gray1)
linesL1 = lsd(gray1)
linesL2 = lsd(gray2)
linesL3 = lsd(gray3)

linesL = np.vstack([linesL1, linesL2])
linesL = np.vstack([linesL, linesL3])
#linesL.extend(linesL3)


for line in linesL:
    x1, y1, x2, y2 = map(int,line[:4])
    dx = x2 - x1
    dy = y2 - y1
    if (dx)**2 + (dy)**2 > 5000:
        #img = cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 3)
        img = cv2.line(img,(x1 - (dx*200),y1 - (dy*200)),(x1 + (dx*200),y1 + (dy*200)),(0,255,0),2)

cv2.namedWindow('window')
cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
