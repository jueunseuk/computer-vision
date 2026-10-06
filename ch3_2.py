# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:23:33 2023

@author: BigData
"""

import cv2 as cv
import matplotlib.pyplot as plt

dim=[2]
img=cv.imread('soccer.jpg')
h=cv.calcHist([img], dim, None, [256], [0, 256])

len(h)
print(type(h))
print(h.shape)

plt.plot(h, color='r', linewidth=1)
plt.show()
