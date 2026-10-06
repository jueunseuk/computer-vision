import cv2 as cv
import sys
import numpy as np

img=cv.imread('C:\junsu\image\soccer.jpg')
if img is None:
    sys.exit("No File exists.")

cv.imshow("Original", img); cv.waitKey()
#%%
hsv=cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('Hsv', hsv) ; cv.waitKey()
#%%
h, s, v = cv.split(hsv)
type(h)
cv.imshow('Hue', h) ; cv.waitKey()
#np.set_printoptions(threshold=np.inf, linewidth=np.inf)
#print(h)
print(np.max(h))

cv.imshow('Sat', s) ; cv.waitKey()
cv.imshow('Value', v) ; cv.waitKey()