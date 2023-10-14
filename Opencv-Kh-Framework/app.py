import cv2
from Framework.ImagePro import Image
from Framework.FaceDet import FaceDetection
from Framework.Mediapipe import Media
from Framework.FeatureDet import Features
import time

img_path = "images/phone.jpg"

# Image.read_img(img_path) # Enter iamge path
Image.display_video(0) # Enter index for webcam or the video path