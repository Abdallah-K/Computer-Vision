import cv2
import time
import mediapipe as mp
import pickle



#####INPUT######
Yaml_file = "Test"
################

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
ptime =0


lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
lbph_recognizer.read(f"Yaml/{Yaml_file}.yml")

labels = {"person_name":1}
with open(f"Yaml/{Yaml_file}.pickle", 'rb' ) as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

seconds=time.time()



while True:
    ret,img = cap.read()


    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4)
    Match =60
    for (x,y,w,h) in faces:
        id,conf = lbph_recognizer.predict(gray[y:y+h,x:x+w])
        if (conf < (1-Match/100)*255):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            name =labels[id]
            per = (1-conf/255)*100
            cv2.putText(img,f"{name} {int(per)}%",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)  
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,"Unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)

        
        

    ctime = time.time()
    fps= 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img,f"{int(fps)}",(20,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1,cv2.LINE_AA)
    cv2.imshow("Recognize",img)
    k = cv2.waitKey(1) &0xFF
    if k ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()