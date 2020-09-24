import cv2
import numpy as np

IMAGE_SIZE_W = 200
def align_image_size(img):
        ratio = IMAGE_SIZE_W/img.shape[1]
        img = cv2.resize(img, (IMAGE_SIZE_W, int(ratio * img.shape[0])))
        return img
# load image, change color spaces, and smoothing
img = cv2.imread('./sample/sample_image.jpg')
img = align_image_size(img)
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_HSV = cv2.GaussianBlur(img_HSV, (9, 9), 3)

# detect tulips
img_H, img_S, img_V = cv2.split(img_HSV)
_thre, img_flowers = cv2.threshold(img_H, 140, 255, cv2.THRESH_BINARY)
cv2.imwrite('tulips_mask.jpg', img_flowers)

# find tulips
contours, hierarchy = cv2.findContours(img_flowers, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )

for i in range(0, len(contours)):
    if len(contours[i]) > 0:

        # remove small objects
        if cv2.contourArea(contours[i]) < 500:
            continue

        cv2.polylines(img, contours[i], True, (255, 255, 255), 5)

# save
cv2.imwrite('result_image.jpg', img)
cv2.namedWindow('window')
cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()  
