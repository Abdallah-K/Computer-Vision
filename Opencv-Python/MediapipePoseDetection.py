import cv2
import mediapipe as mp
import time



mpDraw = mp.solutions.drawing_utils
mppose = mp.solutions.pose
pose=mppose.Pose()



cap = cv2.VideoCapture(0)
ptime = 0

while True:
    _,img = cap.read()

    imgrgb =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results =pose.process(imgrgb)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mppose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h ,w ,c =img.shape
            cx, cy = int(lm.x*w) , int(lm.y*h)#pixel values
            cv2.circle(img,(cx,cy),3,(0,255,0),cv2.FILLED)




    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime =ctime
    cv2.putText(img,f"{int(fps)}",(20,35),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    
    cv2.imshow("Holistic",img)
    key = cv2.waitKey(1) &0xFF
    if key ==ord('q'):
        break