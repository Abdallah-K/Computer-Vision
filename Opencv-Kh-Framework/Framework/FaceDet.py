import cv2
import dlib
import numpy as np

class FaceDetection:
    def __init__(self,image_frame):
        self.img = image_frame
        self.face_cascade = cv2.CascadeClassifier("Framework/models/haarcascade_frontalface_default.xml")
    

    def haar(self,neibrhood=1.3,range = 5,type = True):
        faces = self.face_cascade.detectMultiScale(self.img,neibrhood,range)
        for (x,y,w,h) in faces:
            if type:
                cv2.rectangle(self.img,(x,y),(x+w,y+h),(0,255,0),2)
            else:
                center_coor = x+w //2,y+h//2
                radius = w//2
                cv2.circle(self.img,center_coor,radius,(0,255,0),2)
    

    def deep_learn(self,threshold =0.7):
        h,w,c = self.img.shape
        opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="Framework/models/deploy.prototxt",
                caffeModel="Framework/models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
        
        preprocessed_image = cv2.dnn.blobFromImage(self.img, scalefactor=1.0, size=(300, 300),
                                                mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
    
        opencv_dnn_model.setInput(preprocessed_image)
        results = opencv_dnn_model.forward()    
        for face in results[0][0]:
            face_confidence = face[2]
            if face_confidence > threshold:#threshold
                bbox = face[3:]
                x1 = int(bbox[0] * w)
                y1 = int(bbox[1] * h)
                x2 = int(bbox[2] * w)
                y2 = int(bbox[3] * h)
                cv2.rectangle(self.img,(x1, y1),(x2, y2),(0, 255, 0),2)
    

    def hog(self,threshold = 1):
        hog_face_detector = dlib.get_frontal_face_detector()
        imgrgb=cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        h,w,c = self.img.shape
        results = hog_face_detector(imgrgb, threshold)#threshold
        for bbox in results:
            x1 = bbox.left()
            y1 = bbox.top()
            x2 = bbox.right()
            y2 = bbox.bottom()
            cv2.rectangle(self.img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=w//200)
    


    
    

  

