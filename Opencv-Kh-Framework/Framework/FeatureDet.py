import cv2
from deepface import DeepFace




class Features:
    def __init__(self,image_frame):
        self.img = image_frame
        self.face_cascade = cv2.CascadeClassifier("Framework/models/haarcascade_frontalface_default.xml")
    

    def motion_detection(self,capture):
        _,img2 = capture.read()
        diff = cv2.absdiff(self.img,img2)
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
            cv2.rectangle(self.img,(x,y),(x+w,y+h),(0,255,0),2)
    

    def emotion_detection(self):
        faces=self.face_cascade.detectMultiScale(self.img,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(self.img,(x,y),(x+w,y+h),(0,255,0),2)
            roi =self.img[y:y+h,x:x+w]
            results = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
            cv2.putText(self.img,results['dominant_emotion'],(x,y),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    




