
"""
    图片工具类，提供一些提高图片亮度或者修改尺寸大小的方法
"""
import time
import numpy as np
import cv2



"""
    伽马变化: 可以通过gama变化，提高图片的亮度
"""
def gamma_trans(img, gamma):
    # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    # 实现这个映射用的是OpenCV的查表函数
    return cv2.LUT(img, gamma_table)

"""
    图片尺寸修改，可以统一图片的尺寸
"""
def resizeImg(img):
    dst = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
    return  dst
