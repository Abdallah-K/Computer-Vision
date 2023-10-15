import cv2
import time
import mediapipe as mp


white_color = (255,255,255)
green_color = (0,255,0)
xs = 660
ys = 263
ws = xs + 120
hs = ys + 157
start_video = True
xf = 827
yf = 62
wf = xf + 90
hf = yf + 119
face_cascade_bool = False
face_Media_bool = False
xmf = 823
ymf = 211
wmf = xmf + 98
hmf = ymf + 128
hand_Media_bool = False
xmh = 822
ymh = 386
wmh = xmh + 104
hmh = ymh + 159

def mousePoints(event,x,y,flags,params):
    global xs,ys,ws,hs,start_video,face_cascade_bool,face_Media_bool,xf,yf,wf,hf
    global xmf,ymf,wmf,hmf,xmh,ymh,wmh,hmh,hand_Media_bool
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if x >= xs and x <= ws and y >= ys and y <= hs:
            start_video = not start_video
        elif x >= xf and x <= wf and y >= yf and y <= hf:
            face_cascade_bool = not face_cascade_bool
            face_Media_bool = False
            hand_Media_bool = False
        elif x >= xmf and x <= wmf and y >= ymf and y <= hmf:
            face_Media_bool = not face_Media_bool
            face_cascade_bool = False   
            hand_Media_bool = False
        elif x >= xmh and x <= wmh and y >= ymh and y <= hmh:     
            hand_Media_bool = not hand_Media_bool     
            face_cascade_bool = False
            face_Media_bool = False



cap = cv2.VideoCapture(0)
ptime = 0
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

xv = 10
yv = 106
wv = 650
hv = 586


mphand=mp.solutions.hands
handdetection=mphand.Hands()
mpfacedetection=mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpfacedetection.FaceDetection()

while True:
    _,img = cap.read()
    design = cv2.imread("design.png")

    if start_video:
        design[yv:hv,xv:wv] = img
        cv2.rectangle(design,(xs,ys),(ws,hs),(255,82,140),-1)
        cv2.putText(design,f"Stop",(xs,ys+90),1,cv2.FONT_HERSHEY_COMPLEX,white_color,2)
    else:
        cv2.rectangle(design,(xs,ys),(ws,hs),green_color,-1)
        cv2.putText(design,f"Play",(xs+10,ys+90),1,cv2.FONT_HERSHEY_COMPLEX,white_color,2)

    


    if face_cascade_bool:
        cv2.putText(design[yv:hv,xv:wv],f"Face Detection",(10,50),1,cv2.FONT_HERSHEY_COMPLEX,green_color,2)
        faces = face_cascade.detectMultiScale(design[yv:hv,xv:wv],1.3,4)
        for(x_cascade,y_cascade,w_cascade,h_cascade) in faces:
            cv2.rectangle(design[yv:hv,xv:wv],(x_cascade,y_cascade),(x_cascade+w_cascade,y_cascade+h_cascade),green_color,2)

    
    if face_Media_bool:
        cv2.putText(design[yv:hv,xv:wv],f"FaceDetection Mediapipe",(10,50),1,cv2.FONT_HERSHEY_COMPLEX,green_color,2)
        imgRGB = cv2.cvtColor(design[yv:hv,xv:wv],cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)
        if results.detections:
            for id,detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih , iw ,ic = design[yv:hv,xv:wv].shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                    int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(design[yv:hv,xv:wv],bbox,(0,255,0),2)
    


    if hand_Media_bool:
        cv2.putText(design[yv:hv,xv:wv],f"Hand Detection",(10,50),1,cv2.FONT_HERSHEY_COMPLEX,green_color,2)
        imgrgb=cv2.cvtColor(design[yv:hv,xv:wv],cv2.COLOR_BGR2RGB)
        results = handdetection.process(imgrgb)
        if results.multi_hand_landmarks:
            for handlm in results.multi_hand_landmarks:
                for id, lm in enumerate(handlm.landmark):
                    h,w,c = design[yv:hv,xv:wv].shape
                    cx,cy=int(lm.x*w), int(lm.y*h)
                mpDraw.draw_landmarks(design[yv:hv,xv:wv],handlm,mphand.HAND_CONNECTIONS)




    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(design,f"{int(fps)}",(xs+25,ys-70),1,cv2.FONT_HERSHEY_COMPLEX,white_color,3)

    cv2.imshow("App",design)
    cv2.setMouseCallback("App",mousePoints)
   
    k = cv2.waitKey(1) &0xff
    if k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()