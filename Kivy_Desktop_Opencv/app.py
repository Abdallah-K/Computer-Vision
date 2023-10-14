import kivymd
from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivy.uix.screenmanager import ScreenManager
import cv2
from kivy.core.window import Window
from kivymd.uix.dialog import MDDialog
import os
from PIL import Image
import numpy as np
from kivymd.toast import toast
from kivymd.uix.button import MDFlatButton
import mediapipe as mp
import dlib
from deepface import DeepFace
import time
import face_recognition as fr
import matplotlib.pyplot as plt
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector


Window.size=(400,600)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
id=0
#########
mpfacedetection=mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpfacedetection.FaceDetection()
#####Face Mesh
mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh()
drawSpec=mpDraw.DrawingSpec(thickness=1, circle_radius=2)
#####Hand
mphand=mp.solutions.hands
handdetection=mphand.Hands()
####Pose
mppose = mp.solutions.pose
pose=mppose.Pose()
####HOG
hog_face_detector = dlib.get_frontal_face_detector()
####Deeplearing
opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="models/deploy.prototxt",
                                            caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
####Mouse Event Zones###
draw = False
point = (0,0)
ptx = []
pty =[]
size = 150
#####################


class Manager(ScreenManager):
    pass

class One(Screen):
    pass

class Two(Screen):
    pass

class Three(Screen):
    pass

class Four(Screen):
    pass


class Five(Screen):
    pass


class Six(Screen):
    pass


class Seven(Screen):
    pass

class Eight(Screen):
    pass


class Nine(Screen):
    pass


class Ten(Screen):
    pass


class Eleven(Screen):
    pass


sm =Manager()
sm.add_widget(One(name = "one"))
sm.add_widget(Two(name = "two"))
sm.add_widget(Three(name = "Three"))
sm.add_widget(Four(name = "Four"))
sm.add_widget(Five(name = "five"))
sm.add_widget(Six(name = "six"))
sm.add_widget(Seven(name = "seven"))
sm.add_widget(Eight(name = "eight"))
sm.add_widget(Nine(name = "nine"))
sm.add_widget(Ten(name = "ten"))
sm.add_widget(Eleven(name = "eleven"))

class MyApp(MDApp):
    def build(self):
        self.title="Open-Cv"
        self.theme_cls.primary_palette = "Red"
        return Manager()

    def check_color_ind(self):
        self.theme_cls.primary_palette="LightGreen"
        toast(f"LightGreen")


    def check_color_red(self):
        self.theme_cls.primary_palette="Red"
        toast(f"Red")

    def check_color_purple(self):
        self.theme_cls.primary_palette="DeepPurple"
        toast(f"DeepPurple")
     
    

    def check_color_yellow(self):
        self.theme_cls.primary_palette="Yellow"
        toast(f"Yellow")
     



    def get_yml(self):
        yaml = self.root.ids.two.ids.yml.text
        if yaml == "": 
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Yaml",text="Please enter yaml file name",buttons=[self.close_btn])
            self.dialog.open()
        else:
            path="dataset"
            imagepaths = [os.path.join(path,f) for f in os.listdir(path)]
            faces=[]
            ids=[]
            for imagepath in imagepaths:
                facesim = Image.open(imagepath).convert("L")
                imgnp = np.array(facesim, "uint8")
                id = int(imagepath.split("\\")[1].split("_")[0])
                faces.append(imgnp)
                ids.append(id)
                cv2.waitKey(10)
            
            ids=np.array(ids)
            recognizer.train(faces,ids)
            recognizer.save(f"{yaml}.yml")
            toast(f"{yaml} has been created!")
        

    def dialog(self):
        self.dia=MDDialog(title="AI",text="Demo for a Facial Recognition app")
        self.dia.open()

    def get_cv(self):
        id = self.root.ids.two.ids.generate.text
        if id =="":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Generate",text="Please enter ID for person face",buttons=[self.close_btn])
            self.dialog.open()
        else:
            generatevideo= self.root.ids.two.ids.generatevideo.text
            if generatevideo=="":
                self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
                self.dialog=MDDialog(title="Generate",text="Please enter video",buttons=[self.close_btn])
                self.dialog.open()
            elif generatevideo =="0":
                cap =cv2.VideoCapture(0)
                cout =0
                while True:
                    ret,img = cap.read()
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    for (x,y,w,h) in faces:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        roi = img[y:y+h,x:x+w]
                        cv2.imwrite(f"dataset/{id}_{cout}.jpg",roi)
                        cout+=1

                    if cout>100:
                        break    

                    cv2.imshow("Test",img)
                    k = cv2.waitKey(1) &0xFF
                    if k==ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                cap =cv2.VideoCapture(generatevideo)
                cout =0
                while True:
                    ret,img = cap.read()
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    for (x,y,w,h) in faces:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        roi = img[y:y+h,x:x+w]
                        cv2.imwrite(f"dataset/{id}_{cout}.jpg",roi)
                        cout+=1

                    if cout>100:
                        break    

                    cv2.imshow("Test",img)
                    k = cv2.waitKey(1) &0xFF
                    if k==ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
    
    
    def show_data(self):
        if self.theme_cls.theme_style == "Light":
            self.theme_cls.theme_style="Dark"
        else:
            self.theme_cls.theme_style="Light"
    
    def get_reco(self):
        lbyml=self.root.ids.three.ids.lbphyml.text
        if lbyml == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="LBPH",text="Please enter yaml for person face",buttons=[self.close_btn])
            self.dialog.open()
        else:
            recognizer.read(f"{lbyml}.yml")
            videonamelbph =self.root.ids.three.ids.videonamelbph.text
            if videonamelbph == "":
                self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
                self.dialog=MDDialog(title="LBPH",text="Please enter video please",buttons=[self.close_btn])
                self.dialog.open()
            elif videonamelbph =="0":
                cap =cv2.VideoCapture(0)
                while True:
                    ret,img = cap.read()
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    Match =70
                    for (x,y,w,h) in faces:
                        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                        if (conf < (1-Match/100)*255):
                            if id ==1:
                                id="abdallah"
                                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                                font = cv2.FONT_HERSHEY_DUPLEX
                                per = (1-conf/255)*100
                                cv2.putText(img,f"{str(id)}{int(per)}%",(x,y),font,1,(0,255,0),1,cv2.LINE_AA)
                            else:
                                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                                cv2.putText(img,"Unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)

                    cv2.imshow("LBPH",img)
                    k = cv2.waitKey(1) &0xFF
                    if k ==ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                cap =cv2.VideoCapture(videonamelbph)
                while True:
                    ret,img = cap.read()
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    Match =70
                    for (x,y,w,h) in faces:
                        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                        if (conf < (1-Match/100)*255):
                            if id ==1:
                                id="abdallah"
                                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                                font = cv2.FONT_HERSHEY_DUPLEX
                                per = (1-conf/255)*100
                                cv2.putText(img,f"{str(id)}{int(per)}%",(x,y),font,1,(0,255,0),1,cv2.LINE_AA)
                            else:
                                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                                cv2.putText(img,"Unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)

                    cv2.imshow("LBPH",img)
                    k = cv2.waitKey(1) &0xFF
                    if k ==ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
        
    
    def get_cas(self):
        vidname = self.root.ids.three.ids.videoname.text
        if vidname == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Cascade",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif vidname == "0":
            cap =cv2.VideoCapture(0)
            while True:
                ret,img =cap.read()
                gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces =face_cascade.detectMultiScale(gray,1.3,4)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    
                cv2.imshow("Cascade",img)
                k =cv2.waitKey(1) &0xFF
                if k==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(vidname)
            while True:
                ret,img =cap.read()
                gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces =face_cascade.detectMultiScale(gray,1.3,4)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    
                cv2.imshow("Cascade",img)
                k =cv2.waitKey(1) &0xFF
                if k==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()


    def close_dia(self,obj):
        self.dialog.dismiss()
    

    def media_face(self):
        medaifacevideo = self.root.ids.four.ids.medaifacevideo.text
        if medaifacevideo == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Mediapipe",text="Please enter Video",buttons=[self.close_btn])
            self.dialog.open()
        elif medaifacevideo == "0":
            cap = cv2.VideoCapture(0)
            while True:
                check,img=cap.read()
                imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                results = faceDetection.process(imgRGB)
                if results.detections:
                    for id,detection in enumerate(results.detections):
                        bboxC = detection.location_data.relative_bounding_box
                        ih , iw ,ic = img.shape
                        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                            int(bboxC.width * iw), int(bboxC.height * ih)
                        cv2.rectangle(img,bbox,(0,255,0),2)
                
                cv2.imshow("Face",img)
                key = cv2.waitKey(1) &0xFF
                if key == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(medaifacevideo)
            while True:
                check,img=cap.read()
                imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                results = faceDetection.process(imgRGB)
                if results.detections:
                    for id,detection in enumerate(results.detections):
                        bboxC = detection.location_data.relative_bounding_box
                        ih , iw ,ic = img.shape
                        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                            int(bboxC.width * iw), int(bboxC.height * ih)
                        cv2.rectangle(img,bbox,(0,255,0),2)
                
                cv2.imshow("Face",img)
                key = cv2.waitKey(1) &0xFF
                if key == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
    
    def media_face_mesh(self):
        medaifacemeshvideo = self.root.ids.four.ids.medaifacemeshvideo.text
        if medaifacemeshvideo == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Mediapipe",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif medaifacemeshvideo == "0":
            cap =cv2.VideoCapture(0)
            while True:
                check,img = cap.read()
                imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                results = facemesh.process(imgrgb)
                if results.multi_face_landmarks:
                    for facelms in results.multi_face_landmarks:
                        mpDraw.draw_landmarks(img,facelms)
                
                cv2.imshow("Mesh",img)
                key = cv2.waitKey(1) &0xFF
                if key == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap =cv2.VideoCapture(medaifacemeshvideo)
            while True:
                check,img = cap.read()
                imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                results = facemesh.process(imgrgb)
                if results.multi_face_landmarks:
                    for facelms in results.multi_face_landmarks:
                        mpDraw.draw_landmarks(img,facelms)
                cv2.imshow("Mesh",img)
                key = cv2.waitKey(1) &0xFF
                if key == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
    

    def media_hand(self):
        medaihanddetection = self.root.ids.four.ids.medaihanddetection.text
        if medaihanddetection == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Mediapipe",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif medaihanddetection == "0":
            cap = cv2.VideoCapture(0)
            while True:
                check,img = cap.read()
                imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                results = handdetection.process(imgrgb)
                if results.multi_hand_landmarks:
                    for handlm in results.multi_hand_landmarks:
                        mpDraw.draw_landmarks(img,handlm,mphand.HAND_CONNECTIONS)
                cv2.imshow("Hand",img)
                key = cv2.waitKey(1) &0xFF
                if key==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(medaihanddetection)
            while True:
                check,img = cap.read()
                imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                results = handdetection.process(imgrgb)
                if results.multi_hand_landmarks:
                    for handlm in results.multi_hand_landmarks:
                        mpDraw.draw_landmarks(img,handlm,mphand.HAND_CONNECTIONS)
                cv2.imshow("Hand",img)
                key = cv2.waitKey(1) &0xFF
                if key==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
    

    def media_pose(self):
        mediapose = self.root.ids.four.ids.mediapose.text
        if mediapose == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Mediapipe",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif mediapose == "0":
            cap = cv2.VideoCapture(0)
            while True:
                _,img = cap.read()
                imgrgb =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                results =pose.process(imgrgb)
                if results.pose_landmarks:
                    mpDraw.draw_landmarks(img,results.pose_landmarks,mppose.POSE_CONNECTIONS)
                
                cv2.imshow("Holistic",img)
                key = cv2.waitKey(1) &0xFF
                if key ==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(mediapose)
            while True:
                _,img = cap.read()
                imgrgb =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                results =pose.process(imgrgb)
                if results.pose_landmarks:
                    mpDraw.draw_landmarks(img,results.pose_landmarks,mppose.POSE_CONNECTIONS)
                cv2.imshow("Holistic",img)
                key = cv2.waitKey(1) &0xFF
                if key ==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            

    def hog_detect(self):
        hogvideo = self.root.ids.five.ids.hogvideo.text
        if hogvideo == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="HOG",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif hogvideo =="0":
            cap=cv2.VideoCapture(0)
            while True:
                ret,img=cap.read()
                imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                h,w,c = img.shape
                results = hog_face_detector(imgrgb, 1)#threshold
                for bbox in results:
                    x1 = bbox.left()
                    y1 = bbox.top()
                    x2 = bbox.right()
                    y2 = bbox.bottom()
                    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=w//200)
                
                cv2.imshow("HOG",img)
                k = cv2.waitKey(1) &0xFF
                if k == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap=cv2.VideoCapture(hogvideo)
            while True:
                ret,img=cap.read()
                imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                h,w,c = img.shape
                results = hog_face_detector(imgrgb, 1)#threshold
                for bbox in results:
                    x1 = bbox.left()
                    y1 = bbox.top()
                    x2 = bbox.right()
                    y2 = bbox.bottom()
                    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=w//200)
                
                cv2.imshow("HOG",img)
                k = cv2.waitKey(1) &0xFF
                if k == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()


    def deep_detect(self):
        deepvideo = self.root.ids.five.ids.deepvideo.text
        if deepvideo == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="DeepLearning",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif deepvideo == "0":
            cap = cv2.VideoCapture(0)
            while True:
                ret,img=cap.read()
                h,w,c = img.shape
                preprocessed_image = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300),
                                                            mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
                
                opencv_dnn_model.setInput(preprocessed_image)
                results = opencv_dnn_model.forward()    
                for face in results[0][0]:
                    face_confidence = face[2]
                    if face_confidence > 0.7:#threshold
                        bbox = face[3:]
                        x1 = int(bbox[0] * w)
                        y1 = int(bbox[1] * h)
                        x2 = int(bbox[2] * w)
                        y2 = int(bbox[3] * h)
                        cv2.rectangle(img,(x1, y1),(x2, y2),(0, 255, 0),2)        
                cv2.imshow("Deep learning",img)
                k = cv2.waitKey(1) &0xFF
                if k == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(deepvideo)
            while True:
                ret,img=cap.read()
                h,w,c = img.shape
                preprocessed_image = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300),
                                                            mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
                
                opencv_dnn_model.setInput(preprocessed_image)
                results = opencv_dnn_model.forward()    
                for face in results[0][0]:
                    face_confidence = face[2]
                    if face_confidence > 0.7:#threshold
                        bbox = face[3:]
                        x1 = int(bbox[0] * w)
                        y1 = int(bbox[1] * h)
                        x2 = int(bbox[2] * w)
                        y2 = int(bbox[3] * h)
                        cv2.rectangle(img,(x1, y1),(x2, y2),(0, 255, 0),2)        
                cv2.imshow("Deep learning",img)
                k = cv2.waitKey(1) &0xFF
                if k == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

    def hog_iden(self):
        hogyml = self.root.ids.five.ids.hogyml.text
        if hogyml == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="HoG",text="Please enter yml",buttons=[self.close_btn])
            self.dialog.open()
        else:
            recognizer.read(f"{hogyml}.yml")
            hogrecname = self.root.ids.five.ids.hogrecname.text
            if hogrecname == "":
                self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
                self.dialog=MDDialog(title="HoG",text="Please enter video name",buttons=[self.close_btn])
                self.dialog.open()
            elif hogrecname == "0":
                cap = cv2.VideoCapture(0)
                while True:
                    ret,img=cap.read()
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    h,w,c = img.shape
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    Match =75
                    results = hog_face_detector(imgrgb, 1)#threshold
                    for bbox in results:
                        x1 = bbox.left()
                        y1 = bbox.top()
                        x2 = bbox.right()
                        y2 = bbox.bottom()
                        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0),thickness=2)
                    for (x,y,w,h) in faces:
                        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                        for bbox in results:
                            x1 = bbox.left()
                            y1 = bbox.top()
                            x2 = bbox.right()
                            y2 = bbox.bottom()
                            if (conf < (1-Match/100)*255):
                                if id ==1:
                                    id="abdallah"
                                    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0),thickness=2)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    per = (1-conf/255)*100
                                    cv2.putText(img,f"{str(id)}{int(per)}%",(x1,y1),font,1,(0,255,0),1,cv2.LINE_AA)
                            else:
                                cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
                                font = cv2.FONT_HERSHEY_DUPLEX
                                cv2.putText(img,f"Unkown",(x1,y1),font,1,(0,0,255),1,cv2.LINE_AA)
                    cv2.imshow("HOG",img)
                    k = cv2.waitKey(1) &0xFF
                    if k == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                cap = cv2.VideoCapture(hogrecname)
                while True:
                    ret,img=cap.read()
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    h,w,c = img.shape
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    Match =75
                    results = hog_face_detector(imgrgb, 1)#threshold
                    for bbox in results:
                        x1 = bbox.left()
                        y1 = bbox.top()
                        x2 = bbox.right()
                        y2 = bbox.bottom()
                        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0),thickness=2)
                    for (x,y,w,h) in faces:
                        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                        for bbox in results:
                            x1 = bbox.left()
                            y1 = bbox.top()
                            x2 = bbox.right()
                            y2 = bbox.bottom()
                            if (conf < (1-Match/100)*255):
                                if id ==1:
                                    id="abdallah"
                                    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0),thickness=2)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    per = (1-conf/255)*100
                                    cv2.putText(img,f"{str(id)}{int(per)}%",(x1,y1),font,1,(0,255,0),1,cv2.LINE_AA)
                            else:
                                cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
                                font = cv2.FONT_HERSHEY_DUPLEX
                                cv2.putText(img,f"Unkown",(x1,y1),font,1,(0,0,255),1,cv2.LINE_AA)
                    cv2.imshow("HOG",img)
                    k = cv2.waitKey(1) &0xFF
                    if k == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()

    def deep_face(self):
        deepfacevideo = self.root.ids.six.ids.deepfacevideo.text
        if deepfacevideo == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="DeepFace",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif deepfacevideo == "0":
            cap = cv2.VideoCapture(0)
            while True:
                ret,img = cap.read()
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces=face_cascade.detectMultiScale(gray,1.3,5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    roi =img[y:y+h,x:x+w]
                    results = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
                    cv2.putText(img,results['dominant_emotion'],(x,y),1,cv2.FONT_HERSHEY_COMPLEX,(255,0,0),2)
                cv2.imshow("Deep",img)
                k = cv2.waitKey(30) &0xFF
                if k ==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(deepfacevideo)
            while True:
                ret,img = cap.read()
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces=face_cascade.detectMultiScale(gray,1.3,5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    roi =img[y:y+h,x:x+w]
                    results = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
                    cv2.putText(img,results['dominant_emotion'],(x,y),1,cv2.FONT_HERSHEY_COMPLEX,(255,0,0),2)
                cv2.imshow("Deep",img)
                k = cv2.waitKey(30) &0xFF
                if k ==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

    def atten_detect(self):
        attenyml = self.root.ids.six.ids.attenyml.text
        if attenyml == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Attention",text="Please enter yml",buttons=[self.close_btn])
            self.dialog.open()
        else:
            recognizer.read(f"{attenyml}.yml")
            attenvideo = self.root.ids.six.ids.attenvideo.text
            if attenvideo == "":
                self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
                self.dialog=MDDialog(title="Attention",text="Please enter video",buttons=[self.close_btn])
                self.dialog.open()
            elif attenvideo == "0":
                cap =cv2.VideoCapture(0)
                id=0
                FNT = 0
                AAR = 0
                D1=0
                while True:
                    ret,img = cap.read()
                    FNT+=1
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    Match =75
                    for (x,y,w,h) in faces:
                        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                        if (conf < (1-Match/100)*255):
                            if id ==1:
                                id="abdallah"
                                D1+=1
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            per = (1-conf/255)*100
                            cv2.putText(img,f"{str(id)}{int(per)}%",(x,y),font,1,(0,255,0),1,cv2.LINE_AA)
                        else:
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                            cv2.putText(img,"Unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    AAR=(D1)/(1*FNT)#multiply by number of persons and D1+D2....
                    cv2.putText(img,f"abdallah attention:{round((D1/FNT)*100,2)}%",(20,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    cv2.putText(img,f"Average attention:{round(AAR*100,2)}%",(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    cv2.imshow("Recognize",img)
                    k = cv2.waitKey(1) &0xFF
                    if k ==ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                cap =cv2.VideoCapture(attenvideo)
                id=0
                FNT = 0
                AAR = 0
                D1=0
                while True:
                    ret,img = cap.read()
                    FNT+=1
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    Match =75
                    for (x,y,w,h) in faces:
                        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                        if (conf < (1-Match/100)*255):
                            if id ==1:
                                id="abdallah"
                                D1+=1
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            per = (1-conf/255)*100
                            cv2.putText(img,f"{str(id)}{int(per)}%",(x,y),font,1,(0,255,0),1,cv2.LINE_AA)
                        else:
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                            cv2.putText(img,"Unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    AAR=(D1)/(1*FNT)#multiply by number of persons and D1+D2....
                    cv2.putText(img,f"abdallah attention:{round((D1/FNT)*100,2)}%",(20,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    cv2.putText(img,f"Average attention:{round(AAR*100,2)}%",(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    cv2.imshow("Recognize",img)
                    k = cv2.waitKey(1) &0xFF
                    if k ==ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()


    def deep_atten(self):
        deepattenyml = self.root.ids.six.ids.deepattenyml.text
        if deepattenyml == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="DeepAttention",text="Please enter yaml",buttons=[self.close_btn])
            self.dialog.open()
        else:
            recognizer.read(f"{deepattenyml}.yml")
            deepattenvid = self.root.ids.six.ids.deepattenvid.text
            if deepattenvid =="":
                self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
                self.dialog=MDDialog(title="DeepAttention",text="Please enter video",buttons=[self.close_btn])
                self.dialog.open()
            elif deepattenvid =="0":
                cap = cv2.VideoCapture(0)
                id=0
                FNT = 0
                D1= 0
                # AAR =0
                while True:
                    ret,img = cap.read()
                    FNT+=1
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    Match =75
                    for (x,y,w,h) in faces:
                        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                        roi =img[y:y+h,x:x+w]
                        results = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
                        cv2.putText(img,results['dominant_emotion'],(20,70),1,cv2.FONT_HERSHEY_COMPLEX,(255,0,0),2)
                        if (conf < (1-Match/100)*255):
                            if id ==1:
                                id="abdallah"
                                if results['dominant_emotion'] == "neutral":
                                    D1+=1
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            per = (1-conf/255)*100
                            cv2.putText(img,f"{str(id)}{int(per)}%",(x,y),font,1,(0,255,0),1,cv2.LINE_AA)
                        else:
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                            cv2.putText(img,"Unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    # AAR = (D1)/(1*FNT)
                    cv2.putText(img,f"Attention: {round((D1/FNT)*100,2)}",(20,100),1,cv2.FONT_HERSHEY_COMPLEX,(255,0,0),2)
                    cv2.imshow("Recognize",img)
                    k = cv2.waitKey(1) &0xFF
                    if k ==ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                cap = cv2.VideoCapture(deepattenvid)
                id=0
                FNT = 0
                D1= 0
                # AAR =0
                while True:
                    ret,img = cap.read()
                    FNT+=1
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    Match =75
                    for (x,y,w,h) in faces:
                        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                        roi =img[y:y+h,x:x+w]
                        results = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
                        cv2.putText(img,results['dominant_emotion'],(20,70),1,cv2.FONT_HERSHEY_COMPLEX,(255,0,0),2)
                        if (conf < (1-Match/100)*255):
                            if id ==1:
                                id="abdallah"
                                if results['dominant_emotion'] == "neutral":
                                    D1+=1
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            per = (1-conf/255)*100
                            cv2.putText(img,f"{str(id)}{int(per)}%",(x,y),font,1,(0,255,0),1,cv2.LINE_AA)
                        else:
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                            cv2.putText(img,"Unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    # AAR = (D1)/(1*FNT)
                    cv2.putText(img,f"Attention: {round((D1/FNT)*100,2)}",(20,100),1,cv2.FONT_HERSHEY_COMPLEX,(255,0,0),2)
                    cv2.imshow("Recognize",img)
                    k = cv2.waitKey(1) &0xFF
                    if k ==ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()


    def detect_roi(self,img):
        gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,4)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    def get_roi(self):
        roinb = self.root.ids.seven.ids.roinb.text
        if roinb == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="ROI",text="Please enter roi nb",buttons=[self.close_btn])
            self.dialog.open()
        else:
            roivideo = self.root.ids.seven.ids.roivideo.text
            if roivideo=="":   
                self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
                self.dialog=MDDialog(title="ROI",text="Please enter video",buttons=[self.close_btn])
                self.dialog.open()
            elif roivideo=="0":
                cap = cv2.VideoCapture(0)
                list=[]
                for i in range(0,20):
                    (grabbed,frame1) = cap.read()
                for i in range(int(roinb)):
                    roi = cv2.selectROI(frame1)
                    list.append(roi)
                cv2.destroyAllWindows()
                while True:
                    ret,img = cap.read()
                    for i in range(int(roinb)):
                        roi=list[i]
                        x=int(roi[0])
                        y=int(roi[1])
                        w=int(roi[0]+roi[2])
                        h=int(roi[1]+roi[3])
                        frame3=img[y:h,x:w]
                        self.detect_roi(frame3)
                    cv2.imshow("ROI",img)
                    k = cv2.waitKey(1) &0xFF
                    if k==ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                cap = cv2.VideoCapture(roivideo)
                list=[]
                for i in range(0,20):
                    (grabbed,frame1) = cap.read()
                for i in range(int(roinb)):
                    roi = cv2.selectROI(frame1)
                    list.append(roi)
                cv2.destroyAllWindows()
                while True:
                    ret,img = cap.read()
                    for i in range(int(roinb)):
                        roi=list[i]
                        x=int(roi[0])
                        y=int(roi[1])
                        w=int(roi[0]+roi[2])
                        h=int(roi[1]+roi[3])
                        frame3=img[y:h,x:w]
                        self.detect_roi(frame3)
                    cv2.imshow("ROI",img)
                    k = cv2.waitKey(1) &0xFF
                    if k==ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()

    def get_time(self):
        timernb = self.root.ids.seven.ids.timernb.text
        if timernb =="":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Timer",text="Please enter timer",buttons=[self.close_btn])
            self.dialog.open()
        else:
            timervideo = self.root.ids.seven.ids.timervideo.text
            if timervideo =="":
                self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
                self.dialog=MDDialog(title="Timer",text="Please enter video",buttons=[self.close_btn])
                self.dialog.open()
            elif timervideo =="0":
                cap = cv2.VideoCapture(0)
                sec= time.time()
                while True:
                    ret,img = cap.read()
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    if time.time() - sec>int(timernb):
                        faces = face_cascade.detectMultiScale(gray,1.3,4)
                        for (x,y,w,h) in faces:
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        sec=time.time()
                    cv2.imshow("Timer",img)
                    k = cv2.waitKey(1) &0xFF
                    if k==ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                cap = cv2.VideoCapture(timervideo)
                sec= time.time()
                while True:
                    ret,img = cap.read()
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    if time.time() - sec>int(timernb):
                        faces = face_cascade.detectMultiScale(gray,1.3,4)
                        for (x,y,w,h) in faces:
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        sec=time.time()
                    cv2.imshow("Timer",img)
                    k = cv2.waitKey(1) &0xFF
                    if k==ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()

    def get_fps(self):
        fpsvideo = self.root.ids.seven.ids.fpsvideo.text
        if fpsvideo == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="FPS",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif fpsvideo == "0":
            cap = cv2.VideoCapture(0)
            ptime = 0
            while True:
                ret,img = cap.read()

                ctime =time.time()
                fps = 1/(ctime-ptime)
                ptime=ctime
                cv2.putText(img,f"{int(fps)}",(20,35),1,cv2.FONT_HERSHEY_COMPLEX,(0,0,255),2)
                cv2.imshow("FPS",img)
                k = cv2.waitKey(1) &0xFF
                if k==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(fpsvideo)
            ptime = 0
            while True:
                ret,img = cap.read()

                ctime =time.time()
                fps = 1/(ctime-ptime)
                ptime=ctime
                cv2.putText(img,f"{int(fps)}",(20,35),1,cv2.FONT_HERSHEY_COMPLEX,(0,0,255),2)
                cv2.imshow("FPS",img)
                k = cv2.waitKey(1) &0xFF
                if k==ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

    def get_hog_motion(self):
        hogmotion = self.root.ids.eight.ids.hogmotion.text
        if hogmotion =="":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Hog-Motion",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif hogmotion == "0":
            cap = cv2.VideoCapture(0)
            hog_face_detector = dlib.get_frontal_face_detector()
            class Graph:
                def __init__(self, width, height):
                    self.height = height
                    self.width = width
                    self.graph = np.zeros((height, width, 3), np.uint8)
                def update_frame(self, value):
                    if value < 0:
                        value = 0
                    elif value >= self.height:
                        value = self.height - 1
                    new_graph = np.zeros((self.height, self.width, 3), np.uint8)
                    new_graph[:,:-1,:] = self.graph[:,1:,:]
                    new_graph[self.height - value:,-1,:] = 255
                    self.graph = new_graph
                def get_graph(self):
                    return self.graph
            graph = Graph(100, 60)
            prev_frame = np.zeros((480, 640), np.uint8)
            while True:
                check,img = cap.read()
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                results = hog_face_detector(imgrgb, 1)
                for bbox in results:  
                    x1 = bbox.left()
                    y1 = bbox.top()
                    x2 = bbox.right()     
                    y2 = bbox.bottom()
                    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2),color=(0, 255, 0),thickness=2)
                    gray = cv2.GaussianBlur(gray, (25, 25), None)
                    diff = cv2.absdiff(prev_frame[y1:y2,x1:x2], gray[y1:y2,x1:x2])
                    difference = np.sum(diff)
                    prev_frame = gray
                    graph.update_frame(int(difference/42111))
                    roi = img[-70:-10, -110:-10,:]
                    roi[:] = graph.get_graph()
                cv2.imshow("HOG",img)
                k = cv2.waitKey(1) &0xFF
                if k==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(hogmotion)
            hog_face_detector = dlib.get_frontal_face_detector()
            class Graph:
                def __init__(self, width, height):
                    self.height = height
                    self.width = width
                    self.graph = np.zeros((height, width, 3), np.uint8)
                def update_frame(self, value):
                    if value < 0:
                        value = 0
                    elif value >= self.height:
                        value = self.height - 1
                    new_graph = np.zeros((self.height, self.width, 3), np.uint8)
                    new_graph[:,:-1,:] = self.graph[:,1:,:]
                    new_graph[self.height - value:,-1,:] = 255
                    self.graph = new_graph
                def get_graph(self):
                    return self.graph
            graph = Graph(100, 60)
            prev_frame = np.zeros((480, 640), np.uint8)
            while True:
                check,img = cap.read()
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                results = hog_face_detector(imgrgb, 1)
                for bbox in results:  
                    x1 = bbox.left()
                    y1 = bbox.top()
                    x2 = bbox.right()     
                    y2 = bbox.bottom()
                    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2),color=(0, 255, 0),thickness=2)
                    gray = cv2.GaussianBlur(gray, (25, 25), None)
                    diff = cv2.absdiff(prev_frame[y1:y2,x1:x2], gray[y1:y2,x1:x2])
                    difference = np.sum(diff)
                    prev_frame = gray
                    graph.update_frame(int(difference/42111))
                    roi = img[-70:-10, -110:-10,:]
                    roi[:] = graph.get_graph()
                cv2.imshow("HOG",img)
                k = cv2.waitKey(1) &0xFF
                if k==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()


    def deep_reco_iden(self):
        deeprecognizeyml = self.root.ids.five.ids.deeprecognizeyml.text
        if deeprecognizeyml =="":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="DeepLearning",text="Please enter yml file",buttons=[self.close_btn])
            self.dialog.open()
        else:
            recognizer.read(f"{deeprecognizeyml}.yml")
            deeprecognizevideo = self.root.ids.five.ids.deeprecognizevideo.text
            if deeprecognizevideo =="":
                self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
                self.dialog=MDDialog(title="DeepLearning",text="Please enter video",buttons=[self.close_btn])
                self.dialog.open()
            elif deeprecognizevideo =="0":
                cap=cv2.VideoCapture(0)
                while True:
                    ret,img=cap.read()
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    h,w,c = img.shape

                    preprocessed_image = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300),
                                                                mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
                    Match =75
                    opencv_dnn_model.setInput(preprocessed_image)
                    results = opencv_dnn_model.forward()    
                    for face in results[0][0]:
                        face_confidence = face[2]
                        if face_confidence > 0.7:#threshold
                            bbox = face[3:]
                            x1 = int(bbox[0] * w)
                            y1 = int(bbox[1] * h)
                            x2 = int(bbox[2] * w)
                            y2 = int(bbox[3] * h)
                            cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0),thickness=2)
                            for (x,y,w,h) in faces:
                                id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                                if (conf < (1-Match/100)*255):
                                    if id ==1:
                                        id="abdallah"
                                        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0),thickness=2)
                                        font = cv2.FONT_HERSHEY_DUPLEX
                                        per = (1-conf/255)*100
                                        cv2.putText(img,f"{str(id)}{int(per)}%",(x1,y1),font,1,(0,255,0),1,cv2.LINE_AA)
                                else:
                                    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(img,f"Unkown",(x1,y1),font,1,(0,0,255),1,cv2.LINE_AA)

                    cv2.imshow("Deep learning",img)
                    k = cv2.waitKey(1) &0xFF
                    if k == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            
            else:
                cap=cv2.VideoCapture(deeprecognizevideo)
                while True:
                    ret,img=cap.read()
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    h,w,c = img.shape

                    preprocessed_image = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300),
                                                                mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
                    Match =75
                    opencv_dnn_model.setInput(preprocessed_image)
                    results = opencv_dnn_model.forward()    
                    for face in results[0][0]:
                        face_confidence = face[2]
                        if face_confidence > 0.7:#threshold
                            bbox = face[3:]
                            x1 = int(bbox[0] * w)
                            y1 = int(bbox[1] * h)
                            x2 = int(bbox[2] * w)
                            y2 = int(bbox[3] * h)
                            cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0),thickness=2)
                            for (x,y,w,h) in faces:
                                id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                                if (conf < (1-Match/100)*255):
                                    if id ==1:
                                        id="abdallah"
                                        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0),thickness=2)
                                        font = cv2.FONT_HERSHEY_DUPLEX
                                        per = (1-conf/255)*100
                                        cv2.putText(img,f"{str(id)}{int(per)}%",(x1,y1),font,1,(0,255,0),1,cv2.LINE_AA)
                                else:
                                    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(img,f"Unkown",(x1,y1),font,1,(0,0,255),1,cv2.LINE_AA)

                    cv2.imshow("Deep learning",img)
                    k = cv2.waitKey(1) &0xFF
                    if k == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()


    def get_fr(self):
        frgenerate = self.root.ids.two.ids.frgenerate.text
        if frgenerate =="":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Face-Reconigtion",text="Please enter name",buttons=[self.close_btn])
            self.dialog.open()
        else:
            frgeneratevideo = self.root.ids.two.ids.frgeneratevideo.text
            if frgeneratevideo =="":
                self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
                self.dialog=MDDialog(title="Face-Reconigtion",text="Please enter video",buttons=[self.close_btn])
                self.dialog.open()
            elif frgeneratevideo =="0":
                cap = cv2.VideoCapture(0)
                count = 0
                while True:
                    check,img = cap.read()
                    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
                    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

                    frameloc = fr.face_locations(imgS)
                    encodeframe = fr.face_encodings(imgS,frameloc)

                    for loc in frameloc:
                        y1,x2,y2,x1 = loc
                        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                        save = cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                        path= f'fr-dataset/{frgenerate}.{str(count)}.jpg'
                        count+= 1
                        cv2.imwrite(path,save)
                    if count >5:
                        break
                    cv2.imshow("Dataset",img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                cap = cv2.VideoCapture(frgeneratevideo)
                count = 0
                while True:
                    check,img = cap.read()
                    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
                    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

                    frameloc = fr.face_locations(imgS)
                    encodeframe = fr.face_encodings(imgS,frameloc)

                    for loc in frameloc:
                        y1,x2,y2,x1 = loc
                        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                        save = cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                        path= f'fr-dataset/{frgenerate}.{str(count)}.jpg'
                        count+= 1
                        cv2.imwrite(path,save)
                    if count >5:
                        break
                    cv2.imshow("Dataset",img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()


    def fr_iden(self):
        frvideo = self.root.ids.nine.ids.frvideo.text
        if frvideo =="":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Face-Reconigtion",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif frvideo =="0":
            path='fr-dataset'
            images= []
            classNames= []
            mylist=os.listdir(path)
            for cl in mylist:
                curimg=cv2.imread(f'{path}/{cl}')
                images.append(curimg)
                name = cl.split('.')
                classNames.append(name[0])
            
            encodeList=[]
            for img in images:
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                encode=fr.face_encodings(img)[0]
                encodeList.append(encode)
            cap =cv2.VideoCapture(0)
            while True:
                check,img=cap.read()
                imgS = cv2.resize(img,(0,0),None,0.25,0.25)
                imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
                faceCurframe=fr.face_locations(imgS)
                encodeCurframe=fr.face_encodings(imgS,faceCurframe)
                for encodeface,faceloc in zip(encodeCurframe,faceCurframe):
                    matches=fr.compare_faces(encodeList,encodeface)
                    faceDis = fr.face_distance(encodeList,encodeface)
                    matchindex=np.argmin(faceDis)

                    if matches[matchindex]:
                        name= classNames[matchindex].upper()
                        y1,x2,y2,x1 = faceloc
                        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    else:
                        name="Unkown"
                        y1,x2,y2,x1 = faceloc
                        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

                cv2.imshow("Process",img)
                key = cv2.waitKey(1) &0xFf
                if key == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            path='fr-dataset'
            images= []
            classNames= []
            mylist=os.listdir(path)
            for cl in mylist:
                curimg=cv2.imread(f'{path}/{cl}')
                images.append(curimg)
                name = cl.split('.')
                classNames.append(name[0])
            
            encodeList=[]
            for img in images:
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                encode=fr.face_encodings(img)[0]
                encodeList.append(encode)
            cap =cv2.VideoCapture("720x480.mp4")
            while True:
                check,img=cap.read()
                imgS = cv2.resize(img,(0,0),None,0.25,0.25)
                imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
                faceCurframe=fr.face_locations(imgS)
                encodeCurframe=fr.face_encodings(imgS,faceCurframe)
                for encodeface,faceloc in zip(encodeCurframe,faceCurframe):
                    matches=fr.compare_faces(encodeList,encodeface)
                    faceDis = fr.face_distance(encodeList,encodeface)
                    matchindex=np.argmin(faceDis)

                    if matches[matchindex]:
                        name= classNames[matchindex].upper()
                        y1,x2,y2,x1 = faceloc
                        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    else:
                        name="Unkown"
                        y1,x2,y2,x1 = faceloc
                        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

                cv2.imshow("Process",img)
                key = cv2.waitKey(1) &0xFf
                if key == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()



    def live_plotter(self,x_vec,y1_data,line1,identifier='',pause_time=0.1):
        if line1==[]:
            plt.ion()
            fig = plt.figure(figsize=(5,3))
            ax = fig.add_subplot(111)
            line1, = ax.plot(x_vec,y1_data,'b',alpha=0.8)        
            plt.ylabel('AAR')
            plt.title('Avarage Attention % {}'.format(identifier))
            plt.show()
        
        line1.set_ydata(y1_data)
        if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
            plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
        plt.pause(pause_time)
        return line1


    def average_graph(self):
        averagegraphyml = self.root.ids.eight.ids.averagegraphyml.text
        if averagegraphyml =="":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Average-Graph",text="Please enter yml",buttons=[self.close_btn])
            self.dialog.open()
        else:
            recognizer.read(f"{averagegraphyml}.yml")
            averagegraphvideo = self.root.ids.eight.ids.averagegraphvideo.text
            if averagegraphvideo =="":
                self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
                self.dialog=MDDialog(title="Average-Graph",text="Please enter video name",buttons=[self.close_btn])
                self.dialog.open()
            elif averagegraphvideo =="0":
                cap = cv2.VideoCapture(0)
                id=0
                FNT = 0
                AAR = 0
                D1=0
                size = 200
                x_vec = np.linspace(0,1,size+1)[0:-1]
                y_vec = np.random.randn(len(x_vec))
                line1 = []
                while True:
                    ret,img = cap.read()
                    FNT+=1
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    Match =75
                    for (x,y,w,h) in faces:
                        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                        if (conf < (1-Match/100)*255):
                            if id ==1:
                                id="abdallah"
                                D1+=1
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            per = (1-conf/255)*100
                            cv2.putText(img,f"{str(id)}{int(per)}%",(x,y),font,1,(0,255,0),1,cv2.LINE_AA)
                        else:
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                            cv2.putText(img,"Unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)

                    

                    AAR=(D1)/(1*FNT)#multiply by number of persons and D1+D2....
                    cv2.putText(img,f"abdallah attention:{round((D1/FNT)*100,2)}%",(20,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    cv2.putText(img,f"Average attention:{round(AAR*100,2)}%",(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    cv2.imshow("Recognize",img)

                    y_vec[-1] = AAR*100
                    line1 = self.live_plotter(x_vec,y_vec,line1)
                    y_vec = np.append(y_vec[1:],0.0)

                    k = cv2.waitKey(1) &0xFF
                    if k ==ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
            else:
                cap = cv2.VideoCapture(averagegraphvideo)
                id=0
                FNT = 0
                AAR = 0
                D1=0
                size = 200
                x_vec = np.linspace(0,1,size+1)[0:-1]
                y_vec = np.random.randn(len(x_vec))
                line1 = []
                while True:
                    ret,img = cap.read()
                    FNT+=1
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray,1.3,4)
                    Match =75
                    for (x,y,w,h) in faces:
                        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                        if (conf < (1-Match/100)*255):
                            if id ==1:
                                id="abdallah"
                                D1+=1
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            per = (1-conf/255)*100
                            cv2.putText(img,f"{str(id)}{int(per)}%",(x,y),font,1,(0,255,0),1,cv2.LINE_AA)
                        else:
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                            cv2.putText(img,"Unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)

                    

                    AAR=(D1)/(1*FNT)#multiply by number of persons and D1+D2....
                    cv2.putText(img,f"abdallah attention:{round((D1/FNT)*100,2)}%",(20,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    cv2.putText(img,f"Average attention:{round(AAR*100,2)}%",(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    cv2.imshow("Recognize",img)

                    y_vec[-1] = AAR*100
                    line1 = self.live_plotter(x_vec,y_vec,line1)
                    y_vec = np.append(y_vec[1:],0.0)

                    k = cv2.waitKey(1) &0xFF
                    if k ==ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()


    def hand_count(self):
        handcountervideo = self.root.ids.ten.ids.handcountervideo.text
        if handcountervideo =="":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Hand-Count",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif handcountervideo =="0":
            cap =cv2.VideoCapture(0)
            for i in range(0,20):
                (grabbed,frame1) = cap.read()
            roi = cv2.selectROI(frame1)
            x1=int(roi[0])
            y1=int(roi[1])
            w1=int(roi[0]+roi[2])
            h1=int(roi[1]+roi[3])
            cv2.destroyAllWindows()

            handco=0
            newvlaue1=False
            newvlaue2=True 
            while True:
                ret,img = cap.read()
                frame=img[y1:h1,x1:w1]
                imgrgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = handdetection.process(imgrgb)
                if results.multi_hand_landmarks:
                    newvlaue1=True 
                    for lm in results.multi_hand_landmarks:
                        mpDraw.draw_landmarks(frame,lm,mphand.HAND_CONNECTIONS)
                        if newvlaue1==True and newvlaue2==True:
                            handco=handco+1
                            print(f"abdallah raise his hand {handco} Time")
                            newvlaue2=False
                            newvlaue1=False                                    
                else:
                    newvlaue1=False
                    newvlaue2=True 
                cv2.imshow("Hand",img)
                k = cv2.waitKey(1) &0xFF
                if k==ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            cap =cv2.VideoCapture(handcountervideo)
            for i in range(0,20):
                (grabbed,frame1) = cap.read()
            roi = cv2.selectROI(frame1)
            x1=int(roi[0])
            y1=int(roi[1])
            w1=int(roi[0]+roi[2])
            h1=int(roi[1]+roi[3])
            cv2.destroyAllWindows()

            handco=0
            newvlaue1=False
            newvlaue2=True 
            while True:
                ret,img = cap.read()
                frame=img[y1:h1,x1:w1]
                imgrgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = handdetection.process(imgrgb)
                if results.multi_hand_landmarks:
                    newvlaue1=True 
                    for lm in results.multi_hand_landmarks:
                        mpDraw.draw_landmarks(frame,lm,mphand.HAND_CONNECTIONS)
                        if newvlaue1==True and newvlaue2==True:
                            handco=handco+1
                            print(f"abdallah raise his hand {handco} Time")
                            newvlaue2=False
                            newvlaue1=False                                    
                else:
                    newvlaue1=False
                    newvlaue2=True 
                cv2.imshow("Hand",img)
                k = cv2.waitKey(1) &0xFF
                if k==ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


    def hand_iden(self):
        handcountidenyml =self.root.ids.ten.ids.handcountidenyml.text
        if handcountidenyml =="":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Hand-Iden",text="Please enter yml",buttons=[self.close_btn])
            self.dialog.open()
        else:
            recognizer.read(f"{handcountidenyml}.yml")
            handcountIdentivideo = self.root.ids.ten.ids.handcountIdentivideo.text
            if handcountIdentivideo =="":
                self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
                self.dialog=MDDialog(title="Hand-Iden",text="Please enter video",buttons=[self.close_btn])
                self.dialog.open()
            elif handcountIdentivideo=="0":
                cap = cv2.VideoCapture(0)
                for i in range(0,20):
                    (grabbed,frame1) = cap.read()
                    # frame1 = imutils.resize(frame1,width=1280,height=720)
                roi = cv2.selectROI(frame1)
                x1=int(roi[0])
                y1=int(roi[1])
                w1=int(roi[0]+roi[2])
                h1=int(roi[1]+roi[3])
                cv2.destroyAllWindows()
                handco=0
                newvlaue1=False
                newvlaue2=True 
                while True:
                    ret,img = cap.read()
                    frame=img[y1:h1,x1:w1]
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    imgrgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    results = handdetection.process(imgrgb)
                    resultspose =pose.process(imgrgb)
                    Match = 65
                    if results.multi_hand_landmarks:
                        newvlaue1=True 
                        for lm in results.multi_hand_landmarks:
                            mpDraw.draw_landmarks(frame,lm,mphand.HAND_CONNECTIONS)
                            if resultspose.pose_landmarks:
                                mpDraw.draw_landmarks(frame,resultspose.pose_landmarks,mppose.POSE_CONNECTIONS)
                                faces=face_cascade.detectMultiScale(gray,1.3,4)
                                for (x,y,w,h) in faces:
                                    id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                                    if (conf < (1-Match/100)*255):
                                        if id ==1:
                                            id="Abdallah"
                                            if newvlaue1==True and newvlaue2==True:
                                                handco=handco+1
                                                print(f"{str(id)} raise his hand {handco} Time")
                                                newvlaue2=False
                                                newvlaue1=False
                                                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                                                font = cv2.FONT_HERSHEY_DUPLEX
                                                per = (1-conf/255)*100
                                                cv2.putText(frame,f"{str(id)}{int(per)}%",(x,y),font,1,(0,255,0),1,cv2.LINE_AA)
                                    else:
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                                        cv2.putText(frame,"Unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    else:
                        newvlaue1=False
                        newvlaue2=True                 
                    cv2.imshow("Hand",img)
                    k = cv2.waitKey(1) &0xFF
                    if k==ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
            else:
                cap = cv2.VideoCapture(handcountIdentivideo)
                for i in range(0,20):
                    (grabbed,frame1) = cap.read()
                roi = cv2.selectROI(frame1)
                x1=int(roi[0])
                y1=int(roi[1])
                w1=int(roi[0]+roi[2])
                h1=int(roi[1]+roi[3])
                cv2.destroyAllWindows()
                handco=0
                newvlaue1=False
                newvlaue2=True 
                while True:
                    ret,img = cap.read()
                    frame=img[y1:h1,x1:w1]
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    imgrgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    results = handdetection.process(imgrgb)
                    resultspose =pose.process(imgrgb)
                    Match = 65
                    if results.multi_hand_landmarks:
                        newvlaue1=True 
                        for lm in results.multi_hand_landmarks:
                            mpDraw.draw_landmarks(frame,lm,mphand.HAND_CONNECTIONS)
                            if resultspose.pose_landmarks:
                                mpDraw.draw_landmarks(frame,resultspose.pose_landmarks,mppose.POSE_CONNECTIONS)
                                faces=face_cascade.detectMultiScale(gray,1.3,4)
                                for (x,y,w,h) in faces:
                                    id,conf = recognizer.predict(gray[y:y+h,x:x+w])
                                    if (conf < (1-Match/100)*255):
                                        if id ==1:
                                            id="Abdallah"
                                            if newvlaue1==True and newvlaue2==True:
                                                handco=handco+1
                                                print(f"{str(id)} raise his hand {handco} Time")
                                                newvlaue2=False
                                                newvlaue1=False
                                                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                                                font = cv2.FONT_HERSHEY_DUPLEX
                                                per = (1-conf/255)*100
                                                cv2.putText(frame,f"{str(id)}{int(per)}%",(x,y),font,1,(0,255,0),1,cv2.LINE_AA)
                                    else:
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                                        cv2.putText(frame,"Unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                    else:
                        newvlaue1=False
                        newvlaue2=True                 
                    cv2.imshow("Hand",img)
                    k = cv2.waitKey(1) &0xFF
                    if k==ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
            


    def move(self):
        self.root.ids.three.manager.current ="two"
        self.root.ids.three.manager.transition.direction = "down"


    def cvzone_face(self):
        cvzonefacevideo = self.root.ids.eleven.ids.cvzonefacevideo.text
        if cvzonefacevideo=="":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Cvzone",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif cvzonefacevideo =="0":
            cap = cv2.VideoCapture(0)
            detector = FaceDetector()
            while True:
                success, img = cap.read()
                detector.findFaces(img)
                cv2.imshow("Image", img)
                k = cv2.waitKey(1)
                if k ==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(cvzonefacevideo)
            detector = FaceDetector()
            while True:
                success, img = cap.read()
                detector.findFaces(img)
                cv2.imshow("Image", img)
                k = cv2.waitKey(1)
                if k ==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            

    def cvzone_hand(self):
        cvzonehandvideo = self.root.ids.eleven.ids.cvzonehandvideo.text
        if cvzonehandvideo =="":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Cvzone",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif cvzonehandvideo=="0":
            cap = cv2.VideoCapture(0)
            detector = HandDetector(detectionCon=0.8,maxHands=2)
            while True:
                check,img = cap.read()
                detector.findHands(img)

                cv2.imshow("hand",img)
                k = cv2.waitKey(1) &0xFF
                if k ==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(cvzonehandvideo)
            detector = HandDetector(detectionCon=0.8,maxHands=2)
            while True:
                check,img = cap.read()
                detector.findHands(img)

                cv2.imshow("hand",img)
                k = cv2.waitKey(1) &0xFF
                if k ==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()


    def draw_shape(self,event,x,y,flags,params):
        global point,draw,size
        if event ==cv2.EVENT_LBUTTONDOWN:
            width = x-size
            height = y-size
            if width > 0 and height > 0:
                draw = True
                point=(x,y)
                ptx.append(x)
                pty.append(y)


    def get_mouse(self):
        mousevideo = self.root.ids.seven.ids.mousevideo.text
        if mousevideo == "":
            self.close_btn=MDFlatButton(text="close",on_press=self.close_dia)
            self.dialog=MDDialog(title="Mouse-Event-Zone",text="Please enter video",buttons=[self.close_btn])
            self.dialog.open()
        elif mousevideo=="0":
            cap = cv2.VideoCapture(0)
            cv2.namedWindow("Frame")
            cv2.setMouseCallback("Frame",self.draw_shape)
            while True:
                ret,img = cap.read()
                if draw:
                    for i in range(len(ptx)):
                        cv2.rectangle(img,(ptx[i]-size,pty[i]-size),(ptx[i]+size,pty[i]+size),(0,255,255),2)
                        roi = img[pty[i]-size:pty[i]+size,ptx[i]-size:ptx[i]+size]
                        self.detect_roi(roi)
                cv2.imshow("Frame",img)
                key = cv2.waitKey(1) &0xFF
                if key==ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(mousevideo)
            cv2.namedWindow("Frame")
            cv2.setMouseCallback("Frame",self.draw_shape)
            while True:
                ret,img = cap.read()
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                if draw:
                    for i in range(len(ptx)):
                        cv2.rectangle(img,(ptx[i]-size,pty[i]-size),(ptx[i]+size,pty[i]+size),(0,255,255),2)
                        roi = img[pty[i]-size:pty[i]+size,ptx[i]-size:ptx[i]+size]
                        self.detect_roi(roi)
                cv2.imshow("Frame",img)
                key = cv2.waitKey(1) &0xFF
                if key==ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


MyApp().run()


