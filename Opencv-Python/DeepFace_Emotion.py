import cv2
import time
from deepface import DeepFace


face_cascade= cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
ptime = 0

while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi =img[y:y+h,x:x+w]
        results = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
        cv2.putText(img,results['dominant_emotion'],(x,y),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)

    ctime=time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,f"{int(fps)}",(20,35),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)

    cv2.imshow("Deep",img)
    k = cv2.waitKey(30) &0xFF
    if k ==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()