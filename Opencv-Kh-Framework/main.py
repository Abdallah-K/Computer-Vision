import cv2
from Framework.ImagePro import Image
from Framework.FaceDet import FaceDetection
from Framework.Mediapipe import Media
from Framework.FeatureDet import Features
from Framework.TrackingObj import Track
import time


cap = cv2.VideoCapture(0)
ptime = 0

###Mediapipe###########
# mpDraw = Media.media_draw() 
# faceDetection = Media.face_solution()
# facemesh = Media.mesh_solution()
# handdetection,mphand = Media.hand_solution()
# mppose,pose = Media.pose_solution()
#############################################

##Make the ROi
# list = Image.select_roi(cap)


### Tracking API
# tracker = cv2.TrackerMIL_create()
# tracker = cv2.TrackerKCF_create()
# tracker = cv2.TrackerCSRT_create()
###Tracking
# ok,bbox = Track.track_solution(cap,tracker)


while True:
    ret,img = cap.read()
    image = Image(img)
    image.infinity_video(cap)#Restart the video again
    Faceimage = FaceDetection(img)
    MediaImage = Media(img)

    #Features Detection
    # Features(img).motion_detection(cap)
    # Features(img).emotion_detection()
    ################################

    ###Track Detection
    # Track(img).track_detect(tracker,ok,bbox)


    #Draw the Selected ROI
    # roi_frame = image.draw_roi(list)

    ##Mediapipe Detections###############
    # MediaImage.media_face(faceDetection)
    # MediaImage.media_mesh(facemesh)
    # MediaImage.media_hand(handdetection,mphand)
    # MediaImage.media_pose(pose,mppose)
    ##############################
   
    
    
    #Face Detection Methods###########
    # Faceimage.haar()
    # Faceimage.deep_learn()
    # Faceimage.hog()
    ##############################


    #Image processing##########
    # gray = image.bgr_gray()
    # rgb = image.bgr_rgb()
    # blur = image.blur()
    # canny = image.canny()
    # res = image.resize(300,200)#width,height
    # image.draw_rectangle(10,20,30,40) #draw rectangle
    ##############################
    

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    image.put_text(int(fps),20,35)


    cv2.imshow("Render",img)
    k = cv2.waitKey(1) &0xFF
    if k ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()