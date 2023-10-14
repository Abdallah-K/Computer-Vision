import cv2
import mediapipe as mp
import time



cap = cv2.VideoCapture(0)
ptime = 0
orifps=int(cap.get(cv2.CAP_PROP_FPS))

mpDraw = mp.solutions.drawing_utils
mphand=mp.solutions.hands
handdetection=mphand.Hands()

while True:
    check,img = cap.read()

    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = handdetection.process(imgrgb)


    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            for id, lm in enumerate(handlm.landmark):
                h,w,c = img.shape
                cx,cy=int(lm.x*w), int(lm.y*h)
            mpDraw.draw_landmarks(img,handlm,mphand.HAND_CONNECTIONS)


    ctime = time.time()
    fps = 1/(ctime -ptime)
    ptime = ctime
    cv2.putText(img,f"{int(fps)}",(20,35),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    cv2.imshow("Hand",img)

    key = cv2.waitKey(orifps) &0xFF
    if key==ord('q'):
        break