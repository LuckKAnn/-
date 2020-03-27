import sys

from PyQt5.QtWidgets import QApplication
"""
   行李分拣系统，运行main方法
"""
import os
from UiMain import UI_Main
import multiprocessing
import tensorflow as tf

if  not tf.test.gpu_device_name():
    # 使用CPU
    print("------------------------------正在使用CPU运行-----------------------------------")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    print("------------------------------正在使用GPU运行-----------------------------------")

# What model to download.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if __name__ == "__main__":
    try:
        path = r'./resources/img/first/'
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
        multiprocessing.freeze_support()
        app = QApplication(sys.argv)
        mainWindow =UI_Main()
        mainWindow.showMaximized()
        sys.exit(app.exec_())
    except Exception as E:
        print(E)
