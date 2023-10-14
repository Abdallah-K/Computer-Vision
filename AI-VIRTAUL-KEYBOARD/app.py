import cv2
import mediapipe as mp
import math
import time

cap = cv2.VideoCapture(0)

keys = [["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",",".","/"]]


mpDraw = mp.solutions.drawing_utils
mphand=mp.solutions.hands
handdetection=mphand.Hands()

class Button():
    def __init__(self,pos,text,size):
        self.pos = pos
        self.size = size
        self.text = text

pri_color = (7,174,246)
pri_color_hover = (0,150,215)
pri_color_click = (0,215,129)

def drawALL(img,buttonList):
    for button in buttonList:
        x,y = button.pos
        w,h = button.size
        cv2.rectangle(img,button.pos,(x+w,y+h),pri_color,-1)
        cv2.putText(img, button.text,(x + 20,y + 40),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
    
    return img

x_btn = 30
y_btn = 200
w_btn = x_btn + 50
h_btn = y_btn + 50
buttonList = []
for i in range(len(keys)):
    for j,key in enumerate(keys[i]):
        buttonList.append(Button([60 * j + 25 ,60 * i + 270],key,[55,55]))

screen_text = ""

def msg_box(img,text):
    cv2.rectangle(img,(20,10),(20 + 260,10 + 80),pri_color,-1)
    cv2.putText(img, f"{text}",(30, 70),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),5) 


click_threshold = 15

while True:
    ret,img = cap.read()
    img = cv2.flip(img,1)
    img = cv2.resize(img,(640,480),cv2.INTER_LINEAR)
    h_img,w_img,_ = img.shape


    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = handdetection.process(imgrgb)

    img = drawALL(img,buttonList)

    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handlm,mphand.HAND_CONNECTIONS)
            point_8 = handlm.landmark[8]
            point_4 = handlm.landmark[4]
            cx_8,cy_8 = int(point_8.x * w_img), int(point_8.y * h_img)
            cx_4,cy_4 = int(point_4.x * w_img), int(point_4.y * h_img)
            cv2.circle(img,(cx_8,cy_8),4,(0,255,0),3)
            cv2.circle(img,(cx_4,cy_4),4,(0,255,0),3)
            distance = math.sqrt((cx_4 - cx_8)**2 + (cy_4 - cy_8)**2)
            for button in buttonList:
                x_hover,y_hover = button.pos
                w_hover,h_hover = button.size
                if x_hover < cx_8 < x_hover + w_hover and y_hover < cy_8 < y_hover + h_hover:
                    cv2.rectangle(img,button.pos,(x_hover+w_hover,y_hover+h_hover),pri_color_hover,-1)
                    cv2.putText(img, button.text,(x_hover + 20,y_hover + 40),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                    if distance <= click_threshold:
                        cv2.rectangle(img,button.pos,(x_hover+w_hover,y_hover+h_hover),pri_color_click,-1)
                        cv2.putText(img, button.text,(x_hover + 20,y_hover + 40),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                        screen_text += button.text
                        time.sleep(0.15)
            

    

    cv2.rectangle(img,(50,180),(590,260),pri_color,-1)
    cv2.putText(img, screen_text,(60,240),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),5)





    cv2.rectangle(img,(530,10),(630,110),pri_color,-1)    
    cv2.rectangle(img,(420,10),(520,110),pri_color,-1)    

    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handlm,mphand.HAND_CONNECTIONS)
            point_8 = handlm.landmark[8]
            point_4 = handlm.landmark[4]
            cx_8,cy_8 = int(point_8.x * w_img), int(point_8.y * h_img)
            cx_4,cy_4 = int(point_4.x * w_img), int(point_4.y * h_img)
            cv2.circle(img,(cx_8,cy_8),4,(0,255,0),3)
            cv2.circle(img,(cx_4,cy_4),4,(0,255,0),3)
            distance = math.sqrt((cx_4 - cx_8)**2 + (cy_4 - cy_8)**2)
            if 530 < cx_8 < 630 and 10 < cy_8 < 110:
                cv2.rectangle(img,(530,10),(630,110),pri_color_hover,-1)  
                msg_box(img,"PRINT")
                if distance <= click_threshold:
                    cv2.rectangle(img,(530,10),(630,110),pri_color_click,-1) 
                    if len(screen_text) == 0:
                        msg_box(img,"EMPTY")
                    else:
                        file = open("data.txt","w")
                        file.write(str(screen_text))    
                        file.close()
            
            elif 420 < cx_8 < 520 and 10 < cy_8 < 110:
                cv2.rectangle(img,(420,10),(520,110),pri_color_hover,-1)  
                msg_box(img,"CLEAR")
                if distance <= click_threshold:
                    cv2.rectangle(img,(420,10),(520,110),pri_color_click,-1)
                    screen_text = ""  




    cv2.imshow("Virtual Keyboard",img)
    k = cv2.waitKey(1) &0xff
    if k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()