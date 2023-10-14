import cv2


class Image:
    def __init__(self,image_frame):
        self.img = image_frame
    
    def bgr_gray(self):
        gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        return gray
    

    def bgr_rgb(self):
        rgb = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        return rgb
    

    def blur(self,odd_value_one =9,odd_value_two=9):
        imgBLur=cv2.GaussianBlur(self.img,(odd_value_one,odd_value_two),0)
        return imgBLur
    

    def canny(self,canny_value_one =100,canny_value_two=100):
        imgCanny=cv2.Canny(self.img,canny_value_one,canny_value_two)
        return imgCanny

    
    def infinity_video(self,caputre):
        if caputre.get(cv2.CAP_PROP_POS_FRAMES) == caputre.get(cv2.CAP_PROP_FRAME_COUNT):
            caputre.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    def put_text(self,text,x,y):
        cv2.putText(self.img,f"{text}",(x,y),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),2)
    

    def draw_rectangle(self,x,y,w,h):
        cv2.rectangle(self.img,(x,y),(x+w,y+h),(0,255,0),2)
    
    def resize(self,width,height):
        imgresize=cv2.resize(self.img,(width,height))
        return imgresize

    def select_roi(capture,zones = 1):
        list=[]
        for i in range(0,20):
            (grabbed,frame) = capture.read()
        for i in range(int(zones)):
            roi = cv2.selectROI(frame)
            # print(roi)
            list.append(roi)
        cv2.destroyAllWindows()
        return list
    

    def draw_roi(self,list,zones = 1):
        for i in range(int(zones)):
            roi=list[i]
            x=int(roi[0])
            y=int(roi[1])
            w=int(roi[0]+roi[2])
            h=int(roi[1]+roi[3])
            roi_frame=self.img[y:h,x:w]
            cv2.rectangle(self.img,(x,y),(w,h),(0,255,0),2)
            return roi_frame
    

    def read_img(image_path):
        img = cv2.imread(image_path)
        cv2.imshow("Image",img)
        cv2.waitKey(0)


    def display_video(video_path):
        if video_path ==0:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)
        while True:
            ret,img = cap.read()

            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES,0)

            cv2.imshow("Capture",img)
            k = cv2.waitKey(1)
            if k ==ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()



