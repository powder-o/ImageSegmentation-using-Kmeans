# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 11:24:04 2023

@author: dhruv
"""

import numpy as np 
import cv2

img = cv2.imread('C:/Users/dhruv/.spyder-py3/Machine Learning sreeni/K_Means/xyz.jpg')

img2 = img.reshape((-1,3))
img2 = np.float32(img2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#clusters
k=2 
attempts=10

ret, label, center = cv2.kmeans(img2, k, None, criteria, 
                                attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)

res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imwrite('segmented2.jpg', res2 )



