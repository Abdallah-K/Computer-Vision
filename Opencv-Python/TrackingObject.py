import cv2
import time

# Enter t when you want to choose the tracking object when you finish enter space

ptime = 0
tracker = cv2.TrackerKCF_create()
video = cv2.VideoCapture(0)
cou=1

while True:
    k,frame = video.read()
    cv2.imshow("Tracking",frame)
    k = cv2.waitKey(30) & 0xff
    if k == ord('t'):
        break
bbox = cv2.selectROI(frame, False)

ok = tracker.init(frame, bbox)
cv2.destroyWindow("ROI selector")

while True:
    ok, frame = video.read()
    ok, bbox = tracker.update(frame)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]),
              int(bbox[1] + bbox[3]))
        img = cv2.rectangle(frame, p1, p2, (0,255,0), 2, 2)


    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img,f"{int(fps)}",(20,35),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)


    cv2.imshow("Tracking", frame)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'): 
        break

video.release()
cv2.destroyAllWindows()

