import cv2





class Track:
    def __init__(self,image_frame):
        self.image_frame = image_frame
    

    def track_solution(caputre,tracker):
        ret, frame = caputre.read()
        bbox = cv2.selectROI(frame)
        ok = tracker.init(frame,bbox)
        cv2.destroyAllWindows()
        return ok,bbox
    

    def track_detect(self,tracker,ok,bbox):
        ok,bbox = tracker.update(self.image_frame)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]),
                int(bbox[1] + bbox[3]))
            cv2.rectangle(self.image_frame,p1,p2,(0,255,0),2)
        else:
            cv2.putText(self.image_frame,'Failed to detect',(100,200),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,255,0),3)
