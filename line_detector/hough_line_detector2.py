import cv2
import numpy as np

# カレンダー
img = cv2.imread("./../sample_image2.jpg")
W=200
w_ratio=200/img.shape[1]
img = cv2.resize(img,(200, int(w_ratio*img.shape[0])))
# グレースケール
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
## 反転 ネガポジ変換
gray2 = cv2.bitwise_not(gray)
lines = cv2.HoughLinesP(gray2, rho=1, theta=np.pi/360, threshold=80, minLineLength=200, maxLineGap=5)

for line in lines:
    x1, y1, x2, y2 = line[0]

    # 赤線を引く
    img = cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 3)

cv2.namedWindow('window')
cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
