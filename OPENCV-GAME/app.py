import cv2
import mediapipe as mp
import math
import random

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mphand=mp.solutions.hands
handdetection=mphand.Hands()


speed = 10
score = 0
w_balloon,h_balloon = (100,100)
x_balloon = 0
y_balloon = 0
color_R = (0,255,0)
click_threshold = 20
play_game = False
start_game = False


while True:
    check,img = cap.read()
    img = cv2.resize(img,(640,480),cv2.INTER_LINEAR)
    img = cv2.flip(img,1)
    h_img,w_img,_ = img.shape
    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = handdetection.process(imgrgb)
    cv2.rectangle(img,(0,0),(w_img,h_img),(0,255,0),5)

    # print(h_img,w_img)

    if start_game == False:
        #### Start Button
        color_start = (0,255,0)
        color_exit = (0,255,0)
        rect_width_start = 200
        rect_height_start = 80
        rect_x_start = (w_img - rect_width_start) // 2
        rect_y_start = (h_img - rect_height_start) // 2

        rect_x_exit_start = (w_img - rect_width_start) // 2
        rect_y_exit_start = 300

        if results.multi_hand_landmarks:
            for handlm in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img,handlm,mphand.HAND_CONNECTIONS)
            if len(handlm.landmark) >= 9 and len(handlm.landmark) >= 5:
                point_8 = handlm.landmark[8]
                point_4 = handlm.landmark[4]
                cx_8,cy_8 = int(point_8.x * w_img), int(point_8.y * h_img)
                cx_4,cy_4 = int(point_4.x * w_img), int(point_4.y * h_img)
                cv2.line(img, (cx_8, cy_8), (cx_4, cy_4), (255, 0, 0), 2)
                distance = math.sqrt((cx_4 - cx_8)**2 + (cy_4 - cy_8)**2)     
                if rect_x_start < cx_8 < rect_x_start + rect_width_start and rect_y_start < cy_8 < rect_y_start + rect_height_start:
                    color_start = 255,0,0
                    if distance <= click_threshold:
                        start_game = True
                        play_game = True
                if rect_x_exit_start < cx_8 < rect_x_exit_start + rect_width_start and rect_y_exit_start < cy_8 < rect_y_exit_start + rect_height_start:
                    color_exit = 255,0,0
                    if distance <= click_threshold:
                        break




        start_width = 80
        start_height = 90
        start_x = (w_img - start_width) // 2
        start_y = (h_img - start_height) // 2
        #####
        exit_start_x = (w_img - start_width) // 2
        exit_start_y = 300
        cv2.putText(img, "POP BOX", (160,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.rectangle(img, (rect_x_start, rect_y_start), (rect_x_start + rect_width_start, rect_y_start + rect_height_start), color_start, -1)
        cv2.rectangle(img, (rect_x_exit_start, rect_y_exit_start), (rect_x_exit_start + rect_width_start, rect_y_exit_start + rect_height_start), color_exit, -1)
        cv2.putText(img, "START", (start_x - 10, start_y + start_height // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "EXIT", (exit_start_x, exit_start_y + start_height // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)





    if play_game:
        if results.multi_hand_landmarks:
            for handlm in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img,handlm,mphand.HAND_CONNECTIONS)
            if len(handlm.landmark) >= 9 and len(handlm.landmark) >= 5:
                point_8 = handlm.landmark[8]
                point_4 = handlm.landmark[4]
                cx_8,cy_8 = int(point_8.x * w_img), int(point_8.y * h_img)
                cx_4,cy_4 = int(point_4.x * w_img), int(point_4.y * h_img)
                cv2.line(img, (cx_8, cy_8), (cx_4, cy_4), (255, 0, 0), 2)
                distance = math.sqrt((cx_4 - cx_8)**2 + (cy_4 - cy_8)**2)     
                if x_balloon < cx_8 < x_balloon + w_balloon and y_balloon < cy_8 < y_balloon + h_balloon:
                    color_R = 255,0,0
                    if distance <= click_threshold:
                        score += 1
                        random_y_pop = random.randint(0, (w_img - w_balloon))
                        x_balloon = random_y_pop
                        y_balloon = 0
                        color_R = 0,255,0
                
                else:
                    color_R = 0,255,0



    if play_game:
        cv2.rectangle(img,(x_balloon,y_balloon),(x_balloon + w_balloon,y_balloon + h_balloon),color_R,-1)
        y_balloon += speed 


        if y_balloon == (h_img - h_balloon):
            random_y_drop = random.randint(0, (w_img - w_balloon))
            x_balloon = random_y_drop
            y_balloon = 0
            score -= 1


        cv2.putText(img,f"Score: {score}",(20,50),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)


    if score <= -1:
        # print("GAME OVER")
        play_game = False
        if play_game == False:
            color_retry = (0,255,0)
            color_exit = (0,255,0)
            img_midpt_x = w_img // 2
            img_midpt_y = h_img // 2
            #### RETRY BUTTON
            rect_width = 200
            rect_height = 80
            rect_x = (w_img - rect_width) // 2
            rect_y = (h_img - rect_height) // 2
            #### EXIT BUTTON
            rect_width_exit = 200
            rect_height_exit = 80
            rect_x_exit = (w_img - rect_width_exit) // 2
            rect_y_exit = 300
        
            if results.multi_hand_landmarks:
                for handlm in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img,handlm,mphand.HAND_CONNECTIONS)
                if len(handlm.landmark) >= 9 and len(handlm.landmark) >= 5:
                    point_8 = handlm.landmark[8]
                    point_4 = handlm.landmark[4]
                    cx_8,cy_8 = int(point_8.x * w_img), int(point_8.y * h_img)
                    cx_4,cy_4 = int(point_4.x * w_img), int(point_4.y * h_img)
                    cv2.line(img, (cx_8, cy_8), (cx_4, cy_4), (255, 0, 0), 2)
                    distance = math.sqrt((cx_4 - cx_8)**2 + (cy_4 - cy_8)**2)     
                    if rect_x < cx_8 < rect_x + rect_width and rect_y < cy_8 < rect_y + rect_height:
                        color_retry = 255,0,0
                        if distance <= click_threshold:
                            play_game = True
                            score = 0
                    
                    if rect_x_exit < cx_8 < rect_x_exit + rect_width_exit and rect_y_exit < cy_8 < rect_y_exit + rect_height_exit:
                        color_exit = 255,0,0
                        if distance <= click_threshold:
                            break


            cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), color_retry, -1)
            cv2.rectangle(img, (rect_x_exit,rect_y_exit),(rect_x_exit + rect_width_exit,rect_y_exit + rect_height_exit), color_exit, -1)


            play_width = 80
            play_height = 90
            play_x = (w_img - play_width) // 2
            play_y = (h_img - play_height) // 2
            exit_x = (w_img - play_width) // 2
            exit_y = 300
            cv2.putText(img, "GAME OVER", (160,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(img, "RETRY", (play_x - 10, play_y + play_height // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "EXIT", (exit_x, exit_y + play_height // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow("POP BOX",img)

    key = cv2.waitKey(1) &0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()