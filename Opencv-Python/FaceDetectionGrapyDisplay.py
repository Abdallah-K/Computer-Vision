import cv2
import dlib
import numpy as np



cap = cv2.VideoCapture(0)


hog_face_detector = dlib.get_frontal_face_detector()

class Graph:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.graph = np.zeros((height, width, 3), np.uint8)

    def update_frame(self, value):
        if value < 0:
            value = 0
        elif value >= self.height:
            value = self.height - 1
        new_graph = np.zeros((self.height, self.width, 3), np.uint8)
        new_graph[:,:-1,:] = self.graph[:,1:,:]
        new_graph[self.height - value:,-1,:] = 255
        self.graph = new_graph

    def get_graph(self):
        return self.graph

graph = Graph(100, 60)

prev_frame = np.zeros((480, 640), np.uint8)

while True:
    check,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = hog_face_detector(imgrgb, 1)

    for bbox in results:  
        x1 = bbox.left()
        y1 = bbox.top()
        x2 = bbox.right()     
        y2 = bbox.bottom()
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2),color=(0, 255, 0),thickness=2)
        gray = cv2.GaussianBlur(gray, (25, 25), None)
        diff = cv2.absdiff(prev_frame[y1:y2,x1:x2], gray[y1:y2,x1:x2])
        difference = np.sum(diff)
        prev_frame = gray
        graph.update_frame(int(difference/42111))
        roi = img[-70:-10, -110:-10,:]
        roi[:] = graph.get_graph()


    cv2.imshow("HOG",img)
    k = cv2.waitKey(1) &0xFF
    if k==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()