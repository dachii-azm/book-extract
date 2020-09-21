import cv2
import numpy as np
from pylsd.lsd import lsd

AREA_THRESHOLD =  1000
IMAGE_SIZE_W = 200
LINE_THRESHOLD = 5000

def align_image_size(img):
    ratio = IMAGE_SIZE_W/img.shape[1]
    img = cv2.resize(img, (200, int(ratio * img.shape[0])))
    return img

def detect_lines(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray,(3,3),5)
    gray2 = cv2.GaussianBlur(gray,(5,5),5)
    gray3 = cv2.GaussianBlur(gray,(15,15),3)
    linesL1 = lsd(gray1)
    linesL2 = lsd(gray2)
    linesL3 = lsd(gray3)
    linesL = np.vstack([linesL1, linesL2])
    linesL = np.vstack([linesL, linesL3])
    linesL = np.unique(linesL, axis=0)
    return linesL

def detect_goodLines(linesL):
    goodLines = []
    for line in linesL:
        x1, y1, x2, y2 = map(int,line[:4])
        dx = x2 - x1
        dy = y2 - y1
        if (dx)**2 + (dy)**2 > 5000:
            goodP = [x1, y1, x2, y2]
            goodLines.append(goodP)

    goodLines = get_unique_list(goodLines)
    return goodLines

def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

def inner_product(p1, p2, p3, p4):
    vector1 = np.array([p1[0]-p3[0], p1[1]-p3[1]])
    vector2 = np.array([p2[0]-p4[0], p2[1]-p4[1]])
    inner_pd = np.dot(vector1, vector2)
    return inner_pd

def calc_cross_point(p1, p2, p3, p4):
    cp = [0, 0]
    ref = False
    s1 = ((p4[0]-p2[0])*(p1[1]-p2[1])-(p4[1]-p2[1])*(p1[0]-p2[0]))/2
    s2 = ((p4[0]-p2[0])*(p2[1]-p3[1])-(p4[1]-p2[1])*(p2[0]-p3[0]))/2
    if(abs(s1+s2)>=AREA_THRESHOLD):
        cp = (int(p1[0] + (p3[0]-p1[0])*s1 / (s1+s2)), int(p1[1] + (p3[1]-p1[1])*s1 / (s1+s2)))
        ref = True
    return ref, cp

def get_good_points(goodLines):
    cps = []
    for i in range(len(goodLines)-1):
        for j in range(i+1, len(goodLines)):
            p1 = (goodLines[i][0], goodLines[i][1])
            p3 = (goodLines[i][2], goodLines[i][3])
            p2 = (goodLines[j][0], goodLines[j][1])
            p4 = (goodLines[j][2], goodLines[j][3])
            ref, cp = calc_cross_point(p1, p2, p3, p4)
            if(ref):
                if(cp[0]>=0 and cp[0]<=img.shape[1] and cp[1]>=0 and cp[1]<=img.shape[0] and img[cp[1], cp[0]][2] != 255):
                    cps.append(cp)
                    cv2.circle(img, cp, 2, (0,0,255), 5)
    return cps

def choose_book_edge(cps):
    for i in range(len(cps)-1):
        for j in range(i+1, len(cps)):
            dx = cps[i][0] - cps[j][0]
            dy = cps[i][1] - cps[j][1]    
            dist = np.sqrt((dx)**2 + (dy)**2)
            
            print(dist)
img = cv2.imread('./sample_image.jpg')
img = align_image_size(img)
frame = img.copy()
linesL = detect_lines(img)
goodLines = detect_goodLines(linesL)
cps = get_good_points(goodLines)
print(choose_book_edge(cps))

cv2.namedWindow('window')
cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
