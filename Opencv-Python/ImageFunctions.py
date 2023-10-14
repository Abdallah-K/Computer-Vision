import cv2
import numpy as np


img =cv2.imread("images/controller.png")
kernel=np.ones((5,5),np.uint8)


imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBLur=cv2.GaussianBlur(imgGray,(7,7),0)#odd nbs
imgCanny=cv2.Canny(img,100,100)
imgDialation=cv2.dilate(imgCanny,kernel,iterations=1)
imgEroded=cv2.erode(imgDialation,kernel,iterations=1)


cv2.imshow("Gray",imgGray)
cv2.imshow("blur",imgBLur)
cv2.imshow("Canny",imgCanny)
cv2.imshow("Dialation",imgDialation)
cv2.imshow("Eroded",imgEroded)
cv2.waitKey(0)