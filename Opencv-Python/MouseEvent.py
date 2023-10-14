import cv2


face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

draw = False
point = (0,0)
ptx = []
pty =[]
size = 150
cap = cv2.VideoCapture(0)

def draw_shape(event,x,y,flags,params):
    global point,draw,size
    if event ==cv2.EVENT_LBUTTONDOWN:
        width = x-size
        height = y-size
        if width > 0 and height > 0:
            draw = True
            point=(x,y)
            ptx.append(x)
            pty.append(y)



cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",draw_shape)

def detect(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


while True:
    ret,img = cap.read()
    if draw:
        for i in range(len(ptx)):
            cv2.rectangle(img,(ptx[i]-size,pty[i]-size),(ptx[i]+size,pty[i]+size),(0,255,0),2)
            roi = img[pty[i]-size:pty[i]+size,ptx[i]-size:ptx[i]+size]
            detect(roi)



    cv2.imshow("Frame",img)
    key = cv2.waitKey(1) &0xFF
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()