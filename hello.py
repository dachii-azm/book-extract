import cv2

img = cv2.imread("./sample_image.jpg")
W=200
w_ratio=200/img.shape[1]
img = cv2.resize(img,(200, int(w_ratio*img.shape[0])))

print(img)
