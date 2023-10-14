import cv2
import time
import dlib




cap=cv2.VideoCapture(0)
ptime = 0
opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="models/deploy.prototxt",
                                            caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
sec = time.time()
while True:
    ret,img=cap.read()
    h,w,c = img.shape

    if time.time() - sec >0:
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
                sec = time.time()
            

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img,f"{int(fps)}",(20,35),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)

    cv2.imshow("Deep learning",img)
    k = cv2.waitKey(1) &0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()