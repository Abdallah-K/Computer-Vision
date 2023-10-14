import cv2
import time



cap=cv2.VideoCapture(0)
ptime= 0
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

zones =input("Enter the zones: ")

def detect(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    

list=[]

for i in range(0,20):
    (grabbed,frame1) = cap.read()

for i in range(int(zones)):
    roi = cv2.selectROI(frame1)
    print(roi)
    list.append(roi)
 
cv2.destroyAllWindows()


sec=time.time()

while True:
    _,frame2=cap.read()
    for i in range(int(zones)):
        roi=list[i]
        x=int(roi[0])
        y=int(roi[1])
        w=int(roi[0]+roi[2])
        h=int(roi[1]+roi[3])
        frame3=frame2[y:h,x:w]
        cv2.rectangle(frame2,(x,y),(w,h),(0,255,0),2)
        cv2.cvtColor(frame3,cv2.COLOR_BGR2GRAY)
        detect(frame3)
   

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(frame2,f"{int(fps)}",(20,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("test",frame2)
    k = cv2.waitKey(24) &0xFF
    if k==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()