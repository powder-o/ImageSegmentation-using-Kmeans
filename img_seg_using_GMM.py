# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 21:21:07 2023

@author: dhruv
"""

import numpy as np
import cv2

img = cv2.imread("C:/Users/dhruv/.spyder-py3/Machine Learning sreeni/Image segmentation/xyz.jpg")

img2 = img.reshape((-1,3))

from sklearn.mixture import GaussianMixture as GMM

gmm_model = GMM(n_components=2, covariance_type='tied').fit(img2)
gmm_labels = gmm_model.predict(img2)

orignal_shape = img.shape
segmented = gmm_labels.reshape(orignal_shape[0], orignal_shape[1])
segmented[segmented == 1] = 200
cv2.imwrite("segmentedGMM.jpg", segmented)
