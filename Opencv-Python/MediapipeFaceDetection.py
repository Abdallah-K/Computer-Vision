import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
pTime = 0



mpfacedetection=mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpfacedetection.FaceDetection()


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
            cv2.putText(img,f"score:{int(detection.score[0]* 100)}%",(bbox[0]-20,bbox[1]-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,f"{int(fps)}",(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("Check",img)
    key = cv2.waitKey(1) &0xFF
    if key == ord('q'):
        break