import cv2
import time
import dlib




cap=cv2.VideoCapture(0)
ptime = 0
hog_face_detector = dlib.get_frontal_face_detector()


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


    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img,f"{int(fps)}",(20,35),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)

    cv2.imshow("HOG",img)
    k = cv2.waitKey(1) &0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()