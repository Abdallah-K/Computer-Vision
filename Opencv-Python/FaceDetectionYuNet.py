import cv2
import numpy as np
import time



cap = cv2.VideoCapture(0)


detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2022mar.onnx", "", (320, 320))

def visualize(input, faces, thickness=2):
    FacesLm = True
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            if FacesLm:
                cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)


ptime = 0

while True:
    ret,img = cap.read()


    image_height, image_width, _ = img.shape

    img_W = int(img.shape[1])
    img_H = int(img.shape[0])

    detector.setInputSize((img_W, img_H))

    detections = detector.detect(img)
    visualize(img,detections)


    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img,str(int(fps)),(20,35),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    
    cv2.imshow("Face",img)
    k = cv2.waitKey(1)
    if k ==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()