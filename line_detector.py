import cv2
from pylsd.lsd import lsd

import numpy as np

img = cv2.imread('samp.jpg')
img = cv2.resize(img,(int(img.shape[1]/5),int(img.shape[0]/5)))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),5)
linesL = lsd(gray)

for line in linesL:
    x1, y1, x2, y2 = map(int,line[:4])
    if (x2-x1)**2 + (y2-y1)**2 > 1000:
       # 赤線を引く
       img = cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 3)

cv2.imwrite('samp_pylsd2.jpg',img)
