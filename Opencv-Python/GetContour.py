import cv2
import numpy as np





def getContours(img):
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        cv2.drawContours(imgCon,cnt,-1,(255,0,0),3)
        peri=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.02*peri,True)
        x, y, w, h =cv2.boundingRect(approx)
        cv2.putText(imgCon,"IPhone",
                    (100,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),4)






img = cv2.imread("images/phone.jpg")
imgCon=img.copy()

imgblank=np.zeros_like(img)
Gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(Gray,(7,7),1)
canny=cv2.Canny(blur,50,50)

getContours(canny)

cv2.imshow("Phone",img)
cv2.imshow("Copy",imgCon)

cv2.waitKey(0)