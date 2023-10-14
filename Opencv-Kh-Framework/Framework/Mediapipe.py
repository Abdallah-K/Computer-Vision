import cv2
import mediapipe as mp

class Media:
    def __init__(self,image_frame):
        self.img = image_frame
        self.mpDraw = mp.solutions.drawing_utils
    
    def media_draw():
        mpDraw = mp.solutions.drawing_utils
        return mpDraw
    
    def face_solution():
        mpfacedetection=mp.solutions.face_detection
        faceDetection = mpfacedetection.FaceDetection()
        return faceDetection


    def media_face(self,faceDetection):
        imgRGB = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)
        if results.detections:
            for id,detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih , iw ,ic = self.img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                    int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(self.img,bbox,(0,255,0),2)
                cv2.putText(self.img,f"score:{int(detection.score[0]* 100)}%",(bbox[0]-20,bbox[1]-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)


    def mesh_solution():
        mpfacemesh = mp.solutions.face_mesh
        facemesh = mpfacemesh.FaceMesh()
        return facemesh

    def media_mesh(self,facemesh):
            imgrgb = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
            results = facemesh.process(imgrgb)

            if results.multi_face_landmarks:
                for facelms in results.multi_face_landmarks:
                    self.mpDraw.draw_landmarks(self.img,facelms)
                    for lm in facelms.landmark:
                        ih, iw, ic =self.img.shape
                        x,y =int(lm.x*iw), int(lm.y*ih)
    

    def hand_solution():
        mphand=mp.solutions.hands
        handdetection=mphand.Hands()
        return handdetection,mphand
    

    def media_hand(self,handdetection,mphand):
        imgrgb=cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        results = handdetection.process(imgrgb)

        if results.multi_hand_landmarks:
            for handlm in results.multi_hand_landmarks:
                for id, lm in enumerate(handlm.landmark):
                    h,w,c = self.img.shape
                    cx,cy=int(lm.x*w), int(lm.y*h)
                self.mpDraw.draw_landmarks(self.img,handlm,mphand.HAND_CONNECTIONS)
    

    def pose_solution():
        mppose = mp.solutions.pose
        pose=mppose.Pose()
        return mppose,pose


    def media_pose(self,pose,mppose):
        imgrgb =cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        results =pose.process(imgrgb)

        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(self.img,results.pose_landmarks,mppose.POSE_CONNECTIONS)
            for id,lm in enumerate(results.pose_landmarks.landmark):
                h ,w ,c =self.img.shape
                cx, cy = int(lm.x*w) , int(lm.y*h)#pixel values
                cv2.circle(self.img,(cx,cy),3,(0,255,0),cv2.FILLED)