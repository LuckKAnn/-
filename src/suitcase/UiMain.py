import time
from PyQt5 import QtCore,QtWidgets,QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QGraphicsRectItem, QGraphicsScene, QMainWindow, \
    QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
import cv2
import ReadProcess
import UiBoard
from DatabaseUtils import DbUtils
from multiprocessing import Process, Manager,Queue
from DetectionProcess import DetectionProcess
import os
import shutil

"""
    UI界面主类，通过该类创建UI视图，以及初始化监控和检测进程
"""
class UI_Main(QMainWindow):
    def __init__(self):
        try:
            super(UI_Main, self).__init__()

            self.ui = UiBoard.Ui_MainWindow()
            self.ui.setupUi(self)
            # 设置不同监控区域的显示行
            self.firstRow = int(1)
            self.secondRow = int(1)
            self.thirdRow = int(1)
            # 初始化监控显示表
            self.tableInit()
            # 数据库操作
            self.db = DbUtils()
            # 创建相关队列进行进程间通信
            # 进程读取到图片传入以下队列，通过定时器来拿去图片显示在UI中
            self.firstCameraShow =Queue()
            self.secondCameraShow =Queue()
            self.thirdCameraShow =Queue()
            # 等待进行检测的行李队列，由播放进程假如，由识别进程拿取
            self.waitToDetecte = Queue()
            # 执行读视频进程
            ReadProcess.ReadCamera(self.firstCameraShow, self.secondCameraShow, self.thirdCameraShow, self.waitToDetecte).start()
            # 等到显示行李队列，由识别进程添加，由主进程拿取并显示
            # 同时，识别进程负责找到匹配行李
            self.waitToShow = Queue()
            # 开启识别进程，同时匹配行李
            DetectionProcess(self.waitToDetecte,self.waitToShow).start()

            # 设置定时器，定时从相关队列中取出图片进行显示
            self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
            self.timer_camera.timeout.connect(self.showImage)  # 若定时器结束，则调用show_camera()
            #  初始化定时器，设置定时时间
            self.initTimer()
        except Exception  as  e :
            print(e)


    """
        监控显示表格初始化
        参数: None
        返回值: None
    """
    def tableInit(self):
        # 设置第一监控显示表
        self.ui.tableWidget.setColumnCount(2)
        self.ui.tableWidget.verticalHeader().setVisible(False)  # 隐藏垂直表头
        self.ui.tableWidget.horizontalScrollBar().setVisible(False)
        # self.ui.tableWidget.horizontalHeader().setVisible(False)  # 隐藏水平表头
        self.ui.tableWidget.setHorizontalHeaderLabels(['当前行李', '编号信息'])
        self.ui.tableWidget.setColumnWidth(0,275)
        self.ui.tableWidget.setColumnWidth(1,275)
        # 设置第二监控显示表
        self.ui.tableWidget_2.setColumnCount(4)
        self.ui.tableWidget_2.verticalHeader().setVisible(False)  # 隐藏垂直表头
        self.ui.tableWidget_2.horizontalScrollBar().setVisible(False)
        # self.ui.tableWidget_2.horizontalHeader().setVisible(False)  # 隐藏水平表头
        self.ui.tableWidget_2.setHorizontalHeaderLabels(['当前行李', '匹配图', '匹配信息','分拣口'])

        self.ui.tableWidget_2.setColumnWidth(0, 170)
        self.ui.tableWidget_2.setColumnWidth(1, 170)
        self.ui.tableWidget_2.setColumnWidth(2, 160)
        self.ui.tableWidget_2.setColumnWidth(3, 50)
        # 设置第三监控显示表
        self.ui.tableWidget_3.setColumnCount(4)
        self.ui.tableWidget_3.verticalHeader().setVisible(False)  # 隐藏垂直表头
        self.ui.tableWidget_3.horizontalScrollBar().setVisible(False)
        # self.ui.tableWidget_3.horizontalHeader().setVisible(False)  # 隐藏水平表头
        self.ui.tableWidget_3.setHorizontalHeaderLabels(['当前行李', '匹配图', '匹配信息', '分拣口'])
        self.ui.tableWidget_3.setColumnWidth(0, 170)
        self.ui.tableWidget_3.setColumnWidth(1, 170)
        self.ui.tableWidget_3.setColumnWidth(2, 160)
        self.ui.tableWidget_3.setColumnWidth(3, 50)


    """
        监控信息表添加函数。同时负责随机生成行李的航班信息。
        参数: 
            imageInfo: 监控行李信息，包含图片，匹配图片，匹配行李编号，匹配度
            row: 监控信息表显示行数
            numOfCamera: 监控序号
        返回值: None
    """
    def addImageInfo(self,imageInfo,row,numOfCamera):
        try:
            if numOfCamera == 1:
                ori = imageInfo
                tempUi = self.ui.tableWidget
                tempUi.setRowCount(row)
                row = row - 1
                tempUi.setRowHeight(row, 150)
                show = cv2.resize(imageInfo, (260, 140))
                show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                self.label_pic = QLabel()
                show = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                self.label_pic.setPixmap(QPixmap(show))
                # 数据库存储行李信息
                flightId,flightDst,exit = self.db.generateRandomInfo()
                # 一个行李的编号，由秒级时间戳，加上航班ID，分拣口id，进行编号
                suitcaseNo = str(int(time.time()))+flightId+exit
                # print(suitcaseNo)
                cv2.imwrite(
                    r"./resources/img\first/" + suitcaseNo + ".jpg",
                    ori)
                nameItem = QTableWidgetItem("编号:" + str(suitcaseNo)
                                            +"\n航班信息:"+str(flightId)
                                            + "\n目的地:" + str(flightDst)
                                            + "\n分拣口:" + str(exit))
                self.db.insertSuitcaseInfo(suitcaseNo,flightId,flightDst,exit)
                try:
                    tempUi.setItem(row, 1, nameItem)
                    # 设置行李图片
                    tempUi.setCellWidget(row, 0, self.label_pic)
                except Exception as E:
                    print(E)
            elif numOfCamera== 2:

                ori = imageInfo[0]
                dst = imageInfo[1]
                sno = imageInfo[2]
                print(imageInfo[3])
                print(type(imageInfo[3]))
                point = 100- imageInfo[3]
                point = str("%.3f%%" % (point))

                tempUi = self.ui.tableWidget_2
                tempUi.setRowCount(row)
                row = row - 1
                tempUi.setRowHeight(row, 170)

                show = cv2.resize(ori, (160, 160))
                show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                self.label_pic = QLabel()
                show = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                self.label_pic.setPixmap(QPixmap(show))
                dstshow = cv2.resize(dst, (160, 160))
                dstshow = cv2.cvtColor(dstshow, cv2.COLOR_BGR2RGB)
                self.label_pic2 = QLabel()
                dstshow = QtGui.QImage(dstshow.data, dstshow.shape[1], dstshow.shape[0],
                                       QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                self.label_pic2.setPixmap(QPixmap(dstshow))
                # 通过行李编号查询相关信息
                sno, flightId, flightDst, exit = self.db.querySuitcaseInfo(sno)
                nameItem = QTableWidgetItem("\n匹配度:" + point
                                            + "\n编号:" + sno
                                            + "\n目的地:" + flightDst
                                            + "\n航班号:" + flightId
                                            )
                tempUi.setItem(row, 2, nameItem)
                print("转移监控二当前行李至分拣口:"+exit)
                nameItem = QTableWidgetItem(exit)
                tempUi.setItem(row, 3, nameItem)
                # 设置行李图片
                tempUi.setCellWidget(row, 0, self.label_pic)
                try:
                    tempUi.setCellWidget(row, 1, self.label_pic2)
                except Exception as E:
                    print(E)

            else:
                ori = imageInfo[0]
                dst = imageInfo[1]
                sno = imageInfo[2]
                print(imageInfo[3])
                print(type(imageInfo[3]))
                point = 100 - imageInfo[3]
                point = str("%.3f%%" % (point))
                tempUi = self.ui.tableWidget_3
                tempUi.setRowCount(row)
                row = row - 1
                tempUi.setRowHeight(row, 170)
                # 修剪原图
                show = cv2.resize(ori, (160, 160))
                show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                self.label_pic = QLabel()
                show = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                self.label_pic.setPixmap(QPixmap(show))

                dstshow = cv2.resize(dst, (160, 160))
                dstshow = cv2.cvtColor(dstshow, cv2.COLOR_BGR2RGB)
                self.label_pic2 = QLabel()
                dstshow = QtGui.QImage(dstshow.data, dstshow.shape[1], dstshow.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                self.label_pic2.setPixmap(QPixmap(dstshow))
                # 通过行李编号查询相关信息
                sno,flightId,flightDst,exit = self.db.querySuitcaseInfo(sno)
                nameItem = QTableWidgetItem("\n匹配度:"+point
                                            +"\n编号:" + sno
                                            + "\n目的地:" + flightDst
                                            + "\n航班号:" + flightId
                                            )
                tempUi.setItem(row, 2, nameItem)
                print("转移监控三当前行李至分拣口:" + exit)
                nameItem = QTableWidgetItem(exit)
                tempUi.setItem(row, 3, nameItem)
                # 设置行李图片
                tempUi.setCellWidget(row, 0, self.label_pic)
                tempUi.setCellWidget(row, 1, self.label_pic2)


            tempUi.updateGeometries()
            tempUi.scrollToBottom()
        except Exception as E:
            print(E)
    """
        初始化定时器函数
        参数: None
        返回值:None
    """
    def initTimer(self):
        try:
            if self.timer_camera.isActive() == False:  # 若定时器未启动
                self.timer_camera.start(1)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
            else:
                self.timer_camera.stop()  # 关闭定时器
                self.cap.release()  # 释放视频流
        except Exception as e :
            print(e )



    """
        监控读取图片回显函数，同时负责调用添加监控信息
        在定时器结束时，从响应队列中拿到图片，显示在UI界面上。
        参数: None
        返回值: None
    """
    def showImage(self):
        try:

            # 查询第一监控是否有图片可回显
            if  not self.firstCameraShow.empty():
                value = self.firstCameraShow.get_nowait()
                # cv2.imwrite(r"F:\codesave\python\suitcase04Process\src\suitcase\img\first/"+str(time.time())+".jpg",value)
                show = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                         QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                self.ui.label_1.setPixmap(QPixmap(showImage))

            # 查询第二监控是否有图片可回显
            if not self.secondCameraShow.empty() :
                value = self.secondCameraShow.get_nowait()
                show = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                         QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式

                self.ui.label_2.setPixmap(QPixmap(showImage))

            # 查询第三监控是否有图片可回显
            if not self.thirdCameraShow.empty() :
                value = self.thirdCameraShow.get_nowait()
                show = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                         QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式

                self.ui.label_3.setPixmap(QPixmap(showImage))

            # 查询是否有监控信息待显示
            if not self.waitToShow.empty():
                waitToShowContents = self.waitToShow.get()

                if waitToShowContents[1]==1:
                    self.addImageInfo(waitToShowContents[0],self.firstRow,1)
                    self.firstRow+=1
                elif waitToShowContents[1]==2:
                    self.addImageInfo(waitToShowContents[0],self.secondRow,2)
                    self.secondRow+=1
                else:
                    self.addImageInfo(waitToShowContents[0],self.thirdRow,3)
                    self.thirdRow+=1


        except Exception as E:
            print(E)

