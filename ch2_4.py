# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 19:13:51 2023

@author: BigData
"""

import cv2 as cv
import sys
import numpy as np

# %%
# cap=cv.VideoCapture('slow_traffic_small.mp4')
cap = cv.VideoCapture('C:\junsu\slow_traffic_small.mp4')
if not cap.isOpened():
    sys.exit("연결 실패")

# frams=[]
while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임 획득 실패")
        break
    frame = cv.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
    cv.imshow('Video Display', frame)

    frate = cap.get(cv.CAP_PROP_FPS)

    key=cv.waitKey(int(1000/frate)) # 천천히 재생하도록 만듦
    # key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
print(frate)
cv.destroyAllWindows()
