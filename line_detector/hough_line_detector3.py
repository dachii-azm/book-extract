import numpy as np
import cv2

img = cv2.imread('./../sample_image2.jpg')
W=200
w_ratio=200/img.shape[1]
img = cv2.resize(img,(200, int(w_ratio*img.shape[0])))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,250,apertureSize = 3)
minLineLength = 10
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/360,100,minLineLength,maxLineGap)

for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.namedWindow('window')
cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
