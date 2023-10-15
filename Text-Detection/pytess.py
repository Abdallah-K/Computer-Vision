import cv2
from pytesseract import pytesseract
from pytesseract import Output

pytesseract.tesseract_cmd=r'C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

img = cv2.imread("test.png")


image_data = pytesseract.image_to_data(img,output_type=Output.DICT)
print(image_data['text'])
for i, word in enumerate(image_data['text']):
    if word != "":
        x,y,w,h = image_data['left'][i],image_data['top'][i],image_data['width'][i],image_data['height'][i]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.putText(img,word,(x,y),1,cv2.FONT_HERSHEY_COMPLEX,(0,255,0),1)
        print(word)
print("---------------------------------")
words_in_img = pytesseract.image_to_string(img)
print(words_in_img)  
print("---------------------------------")
letter_boxes = pytesseract.image_to_boxes(img)
print(letter_boxes)
for box in letter_boxes.splitlines():
    box = box.split()
    print(box)



cv2.imshow("TEXT",img)
cv2.waitKey(0)