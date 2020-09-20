import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sample_image.jpg')
W=200
w_ratio=200/img.shape[1]
img = cv2.resize(img,(200, int(w_ratio*img.shape[0])))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.namedWindow('window')
cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
