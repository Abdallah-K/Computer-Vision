import cv2


cap =cv2.VideoCapture(0)


while True:
    ret,img = cap.read()
    ret,img2 = cap.read()
    diff = cv2.absdiff(img,img2)
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    _,threshold = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(threshold,None,iterations=3)
    contours,_ =cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img,contours,-1,(0,255,0),2)
    for c in contours:
        if cv2.contourArea(c) <5000:
            continue
        x,y,w,h =  cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Car",img)
    k = cv2.waitKey(1) &0xFF
    if k==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()