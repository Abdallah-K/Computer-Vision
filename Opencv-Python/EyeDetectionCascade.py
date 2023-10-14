import cv2
import time


cap = cv2.VideoCapture(0)#Webcam or Videoname

face_cascade = cv2.CascadeClassifier("models/haarcascade_eye.xml")
ptime = 0

while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img,f"{int(fps)}",(20,35),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)


    cv2.imshow("Haar",img)
    k = cv2.waitKey(1)
    if k== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()