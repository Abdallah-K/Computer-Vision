import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mphand = mp.solutions.hands
handdetection = mphand.Hands()

x_rec,y_rec,w_rec,h_rec = (20,20,120,120)
color_R = (0,255,0)
color_B = (0,255,0)

while True:
    check,img = cap.read()
    img = cv2.flip(img,1)

    h_img,w_img,_ = img.shape 
    x_box,y_box,w_box,h_box = (w_img - 170,20,150,150)


    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = handdetection.process(imgrgb)


    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handlm,mphand.HAND_CONNECTIONS)
            for id, lm in enumerate(handlm.landmark):
                h,w,c = img.shape
                cx,cy=int(lm.x*w), int(lm.y*h)
                if id == 8:
                    cx_8,cy_8=int(lm.x*w), int(lm.y*h)
                    cv2.circle(img,(cx_8,cy_8),5,(255,0,0),3)
                    if x_rec < cx_8 < x_rec + w_rec and y_rec < cy_8 < y_rec + h_rec:
                        color_R = 255,0,0
                        x_rec = cx_8 - w_rec // 2
                        y_rec = cy_8 - h_rec // 2
                    
                    else:
                        color_R = 0,255,0
    
    
    if x_box < x_rec < x_box + w_box and y_box < y_rec < y_box + h_box:
        color_B = 255,0,0
        cv2.putText(img,"Inside Box",(20,50),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)

    else:
        color_B = 0,255,0



    cv2.rectangle(img,(x_rec,y_rec),(x_rec + w_rec,y_rec + h_rec),color_R,-1)

    cv2.rectangle(img,(x_box,y_box),(x_box + w_box,y_box + h_box),color_B,3)


    cv2.imshow("DRAG DROP",img)

    key = cv2.waitKey(1) &0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()