import cv2
import mediapipe as mp
import pyautogui
import math

cap = cv2.VideoCapture(0)


mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh()
mphand=mp.solutions.hands
handdetection=mphand.Hands()
screen_width,screen_height = pyautogui.size()

while True:
    check,img = cap.read()
    img = cv2.flip(img,1)
    h_img,w_img,_ = img.shape

    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = facemesh.process(imgrgb)
    results_hands = handdetection.process(imgrgb)



    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            point_23 = facelms.landmark[23]
            point_159 = facelms.landmark[159]
            x_23,y_23 =int(point_23.x*w_img), int(point_23.y*h_img)
            x_159,y_159 =int(point_159.x*w_img), int(point_159.y*h_img)
            midpoint_x = (x_23 + x_159) // 2
            midpoint_y = (y_23 + y_159) // 2
            cv2.circle(img, (midpoint_x, midpoint_y), 5, (0, 0, 255), -1)
            index_x = screen_width/w_img * midpoint_x 
            index_y = screen_height/h_img * midpoint_y
            pyautogui.moveTo(index_x,index_y)
            if results_hands.multi_hand_landmarks:
                for handlm in results_hands.multi_hand_landmarks:
                    point_8 = handlm.landmark[8]
                    point_4 = handlm.landmark[4]
                    x_8,y_8 =int(point_8.x*w_img), int(point_8.y*h_img)
                    x_4,y_4 =int(point_4.x*w_img), int(point_4.y*h_img)
                    cv2.circle(img, (x_8,y_8), 5, (0, 0, 255), -1)
                    cv2.circle(img, (x_4,y_4), 5, (0, 0, 255), -1)
                    distance = math.sqrt((x_4 - x_8)**2 + (y_4 - y_8)**2)
                    if distance <= 30:
                        pyautogui.click()     






    cv2.imshow("Mesh",img)
    key = cv2.waitKey(1) &0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()