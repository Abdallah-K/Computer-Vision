import cv2
import sys
sys.stdout.reconfigure(encoding='utf-8')
import easyocr


img = cv2.imread("test.png")


reader = easyocr.Reader(['en'])
ocr_results = reader.readtext(img)

for detection in ocr_results:
    top_left = detection[0][0]
    bottom_right = detection[0][2]
    text_ocr = detection[1]
    cv2.rectangle(img, (top_left[0],top_left[1] + 40), (bottom_right[0] + 63, bottom_right[1]), (0, 255, 0), 2)
    print(text_ocr)



cv2.imshow("TEXT",img)
cv2.waitKey(0)