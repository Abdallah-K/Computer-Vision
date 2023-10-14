import cv2
from pathlib import Path


def saveImage(userid,username,usercounter,roi_img):
    Path(f"Dataset/{userid}").mkdir(parents=True,exist_ok=True)
    cv2.imwrite(f"Dataset/{userid}/{username}_{userid}_{usercounter}.jpg",roi_img)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


#####INPUT######
userid = 0
username = "UserName"
################


usercounter = 0


while True:
    ret,img = cap.read()

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)


    faces = face_cascade.detectMultiScale(img,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi = img[y:y+h,x:x+w]
        saveImage(userid,username,usercounter,roi)
        usercounter +=1
    

    if usercounter >20:
        break



    cv2.imshow("FACE",img)
    k = cv2.waitKey(1)
    if k ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()