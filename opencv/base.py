# -*- coding: UTF-8 -*-
# @Author：MengKang
# @Date：2023/02/02 16:21
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("images/longmao.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
