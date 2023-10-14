import cv2
import mediapipe as mp
import time



cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh()
drawSpec=mpDraw.DrawingSpec(thickness=1, circle_radius=2)


while True:
    check,img = cap.read()

    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = facemesh.process(imgrgb)



    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,facelms)
            for lm in facelms.landmark:
                ih, iw, ic =img.shape
                x,y =int(lm.x*iw), int(lm.y*ih)
                



    cTime =time.time()
    fps = 1/(cTime - pTime)
    pTime =cTime
    cv2.putText(img,f"{int(fps)}",(20,35),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)

    cv2.imshow("Mesh",img)
    key = cv2.waitKey(1) &0xFF
    if key == ord('q'):
        break

