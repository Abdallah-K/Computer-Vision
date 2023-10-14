import cv2


img = cv2.imread("images/controller.png")

h,w,_ = img.shape

grid_x = 2
grid_y = 2


for x in range(0,grid_x):
    for y in range(0,grid_y):
        xratio1 = x/grid_x
        xratio2 = (x+1)/grid_x
        yratio1 = y/grid_x
        yratio2 = (y+1)/grid_y
        part = img[int(yratio1*h):int(yratio2*h),int(xratio1*w):int(xratio2*w)].copy()
        cv2.imshow(f"x = {x},y = {y}",part)



cv2.imshow("Face",img)
cv2.waitKey(0)

