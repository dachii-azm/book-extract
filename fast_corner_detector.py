import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('sample_image.jpg',0)
W=200
w_ratio=200/img.shape[1]
img = cv2.resize(img,(200, int(w_ratio*img.shape[0])))
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# Disable nonmaxSuppression
#fast.setBool('nonmaxSuppression',0)
fast.getThreshold(5)
kp = fast.detect(img,None)

img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

cv2.namedWindow('window')
cv2.imshow('window', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
