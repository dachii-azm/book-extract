import cv2
from pylsd.lsd import lsd
import numpy as np

AREA_THRESHOLD = 1000

img = cv2.imread('./sample_image.jpg')
frame = img.copy()
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
linesL = np.unique(linesL, axis=0)
goodLines = [] 
cps = []

for line in linesL:
    x1, y1, x2, y2 = map(int,line[:4])
    dx = x2 - x1
    dy = y2 - y1
    if (dx)**2 + (dy)**2 > 5000:
        #img = cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 3)
        img = cv2.line(img,(x1 - (dx*200),y1 - (dy*200)),(x1 + (dx*200),y1 + (dy*200)),(0,255,0),2)
        #goodP = [x1 - (dx*200), y1 - (dy*200), x1 + (dx*200), y1 + (dy*200)]
        goodP = [x1, y1, x2, y2]
        goodLines.append(goodP)
        
def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

def inner_product(p1, p2, p3, p4):
    vector1 = np.array([p1[0]-p3[0], p1[1]-p3[1]])
    vector2 = np.array([p2[0]-p4[0], p2[1]-p4[1]])
    inner_pd = np.dot(vector1, vector2)
    return inner_pd

goodLines = get_unique_list(goodLines)

for i in range(len(goodLines)-1):
    for j in range(i+1, len(goodLines)):
        p1 = (goodLines[i][0], goodLines[i][1])
        p3 = (goodLines[i][2], goodLines[i][3])
        p2 = (goodLines[j][0], goodLines[j][1])
        p4 = (goodLines[j][2], goodLines[j][3])

        s1 = ((p4[0]-p2[0])*(p1[1]-p2[1])-(p4[1]-p2[1])*(p1[0]-p2[0]))/2
        s2 = ((p4[0]-p2[0])*(p2[1]-p3[1])-(p4[1]-p2[1])*(p2[0]-p3[0]))/2
        if(abs(s1+s2)>=AREA_THRESHOLD):
            cp = (int(p1[0] + (p3[0]-p1[0])*s1 / (s1+s2)), int(p1[1] + (p3[1]-p1[1])*s1 / (s1+s2)))
            if(cp[0]>=0 and cp[0]<=img.shape[1] and cp[1]>=0 and cp[1]<=img.shape[0] and img[cp[1], cp[0]][2] != 255):
                cps.append(cp)        
                cv2.circle(img, cp, 2, (0,0,255), 5)




cv2.namedWindow('window')
cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
