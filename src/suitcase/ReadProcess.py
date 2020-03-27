# coding: utf-8
import queue
import time
from multiprocessing import Process

import numpy as np
import os
import tensorflow as tf
import  cv2
from object_detection.utils import label_map_util

MODEL_NAME = 'suitcase_detction'
PATH_TO_FROZEN_GRAPH = r'./resources/models/frozen_inference_graph.pb'
PATH_TO_LABELS = r'./resources/models/suitcase_label_map.pbtxt'
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

"""
    读取监控摄像头进程。通过该进程，不停读取视频帧，传入显示队列。
    同时利用轮询的方法，每隔一段时间，识别图片行李，当发现行李在指定位置内时，进行跟踪。
    为确保不重复跟踪同一行李，通过计算中心点是否落在其它跟踪区域内，来避免重复跟踪。最终，当跟踪物体到达某一位置时，截取跟踪区域，传入识别队列，由识别进程再次识别

"""
class ReadCamera(Process):


    def __init__(self,list,list2,list3,listWaitToDetecte):
        super().__init__()
        self.list = list
        self.list2 = list2
        self.list3 = list3
        # 设置轮询的定时器
        self.firstTimes = 20
        self.secondTimes = 20
        self.thirdTimes = 20
        # 设置追踪器
        self.trackersFirst = []
        self.trackersPositionFirst = []
        self.trackersSecond = []
        self.trackersPositionSecond  = []
        self.trackersThird = []
        self.trackersPositionThird = []

        #设置等待检测行李
        self.listWaitToDetecte = listWaitToDetecte

    def run(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')



        # 读取视频
        CAMERA = r"./resources/video/1.mp4"
        CAMERA22 = r"./resources/video/2.mp4"
        CAMERA33 = r"./resources/video/3.mp4"
        capture = cv2.VideoCapture(CAMERA)
        capture2 = cv2.VideoCapture(CAMERA22)
        capture3 = cv2.VideoCapture(CAMERA33)

        if not capture.isOpened():
            print("视频打开失败")
            exit(0)

        tracker = cv2.MultiTracker_create()

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                    ret, image_np = capture.read()
                    ret2, image_np2 = capture2.read()
                    ret3, image_np3 = capture3.read()
                    if image_np is not None:
                        ori1 = self.playFirstCamera(detection_graph,sess,image_np)
                    if image_np2 is not None:
                        ori2 = self.playSecondCamera(detection_graph,sess,image_np2)
                    if image_np3 is not None:
                        ori3 = self.playThirdCamera(detection_graph,sess,image_np3)

                    # 当三个监控都读取完毕时，结束
                    if image_np is None and image_np3 is None and image_np2 is None:
                        break

                    if self.list.qsize()>=100 or self.list2.qsize()>=100 or self.list3.qsize()>=100:
                        time.sleep(1)
                    self.list.put(ori1)
                    self.list2.put(ori2)
                    self.list3.put(ori3)


    """
        读取第二个视频帧，并根据轮询法判断是否进行检测。若检测发现行李，即进行跟踪
    """
    def playSecondCamera(self,detection_graph,sess,img):
        cv2.putText(img, str(self.secondTimes), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255), 2)

        # cv2.rectangle(img, (200, 270), (800, 680), (205, 240, 0), 2)
        if self.secondTimes!=0:
            self.secondTimes-=1
        else:
            image_np_expanded = np.expand_dims(img, axis=0)
            height,width,  a = img.shape
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     img,
            #     np.squeeze(boxes),
            #     np.squeeze(classes).astype(np.int32),
            #     np.squeeze(scores),
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=8,
            #     min_score_thresh=0.9)

            s_boxes = boxes[scores > 0.8]

            for i in range(0, s_boxes.shape[0]):
                ymin = int(s_boxes[i][0] * height)  # ymin
                xmin = int(s_boxes[i][1] * width)  # xmin
                ymax = int(s_boxes[i][2] * height)  # ymax
                xmax = int(s_boxes[i][3] * width)  # xmax
                # print(str(xmin)+" "+str(xmax)+" "+str(ymin)+" "+str(ymax))
                # if xmax>xmin+1 and ymax>ymin+1:
                if ymin >= 270 and ymax <= 680:
                    flag = True
                    midx = int(xmin + (xmax - xmin) / 2)
                    midy = int(ymin + (ymax - ymin) / 2)
                    for i in range(len(self.trackersPositionFirst)):
                        if self.trackersPositionFirst[i][0] < midx < self.trackersPositionFirst[i][2] and \
                                self.trackersPositionFirst[i][1] < midy < \
                                self.trackersPositionFirst[i][3]:
                            flag = False
                            break
                    if flag:
                        tracker = cv2.TrackerCSRT().create()
                        # tracker = cv2.TrackerKCF().create()
                        # tracker = cv2.TrackerMOSSE().create()
                        tracker.init(img, (xmin, ymin, xmax - xmin, ymax - ymin))
                        self.trackersFirst.append(tracker)
                        self.trackersPositionFirst.append((xmin, ymin, xmax, ymax))
            self.secondTimes = 20


        out = queue.Queue()
        # 更新跟踪器
        for i in range(len(self.trackersFirst)):
            tracker = self.trackersFirst[i]
            ok, bboxes = tracker.update(img)
            if ok:
                p1 = (int(bboxes[0]), int(bboxes[1]))
                p2 = (int(bboxes[0] + bboxes[2]), int(bboxes[1] + bboxes[3]))
                self.trackersPositionFirst[i] = (
                    int(bboxes[0]), int(bboxes[1]), int(bboxes[0] + bboxes[2]), int(bboxes[1] + bboxes[3]))
                # cv2.rectangle(img, p1, p2, (0, 0, 255), 2, 10)
                if int(bboxes[1] + bboxes[3]) < 410:
                    ymin = int(bboxes[1]) - 40
                    xmin = int(bboxes[0]) -40
                    ymax = int(bboxes[1] + bboxes[3]) + 40
                    xmax = int(bboxes[0] + bboxes[2]) + 40
                    part = img[ymin:ymax, xmin:xmax]

                    self.listWaitToDetecte.put((part,2))
                    # result = self.detecte2(detection_graph, sess, part)
                    # if result is None:
                    #     print("有行李未被检测出")
                    # else:
                    #     cv2.imwrite(r"./img/first/" + str(time.time()) + ".jpg", result)
                    out.put(i)
                if  int(bboxes[2])*int(bboxes[3])>=150000:
                    print("有行李出现错误，失效")
                    out.put(i)

        while not out.empty():
            outIndex = out.get()
            self.trackersFirst.pop(outIndex)
            self.trackersPositionFirst.pop(outIndex)

        return img

    """
          读取第一个视频帧，并根据轮询法判断是否进行检测。若检测发现行李，即进行跟踪
      """
    def playFirstCamera(self,detection_graph,sess,img):
        # 显示轮询法参数
        cv2.putText(img, str(self.firstTimes), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255), 2)
        # cv2.rectangle(img, (340, 450), (800, 700), (205, 240, 0), 2)
        if self.firstTimes!=0:
            self.firstTimes-=1
        else:
            image_np_expanded = np.expand_dims(img, axis=0)
            height,width,  a = img.shape
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     img,
            #     np.squeeze(boxes),
            #     np.squeeze(classes).astype(np.int32),
            #     np.squeeze(scores),
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=8,
            #     min_score_thresh=0.9)

            #过滤置信度低于80%的目标
            s_boxes = boxes[scores > 0.8]
            for i in range(0, s_boxes.shape[0]):
                ymin = int(s_boxes[i][0] * height)  # ymin
                xmin = int(s_boxes[i][1] * width)  # xmin
                ymax = int(s_boxes[i][2] * height)  # ymax
                xmax = int(s_boxes[i][3] * width)  # xmax
                # print(str(xmin)+" "+str(xmax)+" "+str(ymin)+" "+str(ymax))
                # if xmax>xmin+1 and ymax>ymin+1:
                #设置监控范围在【450，700】的竖直方向
                if ymin >= 450 and ymax <= 700:
                    flag = True
                    midx = int(xmin + (xmax - xmin) / 2)
                    midy = int(ymin + (ymax - ymin) / 2)
                    for i in range(len(self.trackersPositionSecond)):
                        if self.trackersPositionSecond[i][0] < midx < self.trackersPositionSecond[i][2] and self.trackersPositionSecond[i][1] < midy < \
                                self.trackersPositionSecond[i][3]:
                            flag = False
                            break
                    if flag:
                        tracker = cv2.TrackerCSRT().create()
                        # tracker = cv2.TrackerKCF().create()
                        # tracker = cv2.TrackerMOSSE().create()
                        tracker.init(img, (xmin, ymin, xmax - xmin, ymax - ymin))
                        self.trackersSecond.append(tracker)
                        self.trackersPositionSecond.append((xmin, ymin, xmax, ymax))
            self.firstTimes = 15

        out = queue.Queue()
        # 更新跟踪器
        for i in range(len(self.trackersSecond)):
            tracker = self.trackersSecond[i]
            ok, bboxes = tracker.update(img)
            if ok:
                p1 = (int(bboxes[0]), int(bboxes[1]))
                p2 = (int(bboxes[0] + bboxes[2]), int(bboxes[1] + bboxes[3]))
                self.trackersPositionSecond[i] = (
                int(bboxes[0]), int(bboxes[1]), int(bboxes[0] + bboxes[2]), int(bboxes[1] + bboxes[3]))
                # cv2.rectangle(img, p1, p2, (0, 0, 255), 2, 10)
                #判断是否到达结束区域
                if int(bboxes[1] + bboxes[3]) < 550:
                    ymin = int(bboxes[1]) - 15
                    xmin = int(bboxes[0]) - 15
                    ymax = int(bboxes[1] + bboxes[3]) + 15
                    xmax = int(bboxes[0] + bboxes[2]) + 15
                    part = img[ymin:ymax, xmin:xmax]
                    self.listWaitToDetecte.put((part, 1))
                    # result = self.detecte2(detection_graph,sess,part)
                    # if result is None:
                    #     print("有行李未被检测出")
                    # else:
                    #     cv2.imwrite(r"./img/second/" + str(time.time()) + ".jpg", result)
                    out.put(i)
                if int(bboxes[2]) * int(bboxes[3]) >= 150000:
                    print("有行李出现错误，失效")
                    out.put(i)

        while not out.empty():
            outIndex = out.get()
            self.trackersSecond.pop(outIndex)
            self.trackersPositionSecond.pop(outIndex)

        return img

    """
          读取第三个视频帧，并根据轮询法判断是否进行检测。若检测发现行李，即进行跟踪
    """
    def playThirdCamera(self,detection_graph,sess,img):
        # 显示轮询法参数
        cv2.putText(img, str(self.thirdTimes), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255), 2)
        # cv2.rectangle(img, (580, 370), (1050, 680), (205, 240, 0), 2)
        if self.thirdTimes!=0:
            self.thirdTimes-=1
        else:
            image_np_expanded = np.expand_dims(img, axis=0)
            height, width, a = img.shape
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # 可视化工具
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     img,
            #     np.squeeze(boxes),
            #     np.squeeze(classes).astype(np.int32),
            #     np.squeeze(scores),
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=8,
            #     min_score_thresh=0.9)
            #过滤置信度低于80%的行李
            s_boxes = boxes[scores > 0.8]

            for i in range(0, s_boxes.shape[0]):
                ymin = int(s_boxes[i][0] * height)  # ymin
                xmin = int(s_boxes[i][1] * width)  # xmin
                ymax = int(s_boxes[i][2] * height)  # ymax
                xmax = int(s_boxes[i][3] * width)  # xmax
                # print(str(xmin)+" "+str(xmax)+" "+str(ymin)+" "+str(ymax))
                # if xmax>xmin+1 and ymax>ymin+1:
                # 设置监控区域范围 ，在[370,680]竖直区间内
                if ymin >= 370 and ymax <= 680:
                    flag = True
                    midx = int(xmin + (xmax - xmin) / 2)
                    midy = int(ymin + (ymax - ymin) / 2)
                    for i in range(len(self.trackersPositionThird)):
                        if self.trackersPositionThird[i][0] < midx < self.trackersPositionThird[i][2] and \
                                self.trackersPositionThird[i][1] < midy < \
                                self.trackersPositionThird[i][3]:
                            flag = False
                            break
                    if flag:
                        # 初始化追踪器
                        tracker = cv2.TrackerCSRT().create()
                        # tracker = cv2.TrackerKCF().create()
                        # tracker = cv2.TrackerMOSSE().create()
                        tracker.init(img, (xmin, ymin, xmax - xmin, ymax - ymin))
                        self.trackersThird.append(tracker)
                        self.trackersPositionThird.append((xmin, ymin, xmax, ymax))
            self.thirdTimes = 20

        out = queue.Queue()
        # 跟新追踪器
        for i in range(len(self.trackersThird)):
            tracker = self.trackersThird[i]
            ok, bboxes = tracker.update(img)
            if ok:
                p1 = (int(bboxes[0]), int(bboxes[1]))
                p2 = (int(bboxes[0] + bboxes[2]), int(bboxes[1] + bboxes[3]))
                self.trackersPositionThird[i] = (
                    int(bboxes[0]), int(bboxes[1]), int(bboxes[0] + bboxes[2]), int(bboxes[1] + bboxes[3]))
                # cv2.rectangle(img, p1, p2, (0, 0, 255), 2, 10)
                # 判断是否到达结束区域
                if int(bboxes[1] + bboxes[3]) < 470:
                    ymin = int(bboxes[1]) - 20
                    xmin = int(bboxes[0]) - 20
                    ymax = int(bboxes[1] + bboxes[3]) + 20
                    xmax = int(bboxes[0] + bboxes[2]) + 20
                    part = img[ymin:ymax, xmin:xmax]
                    self.listWaitToDetecte.put((part, 3))
                    out.put(i)
                 # 判断面积，当面积太大时，说明追踪算法已经失效，应该舍弃
                if  int(bboxes[2])*int(bboxes[3])>=150000:
                    print("有行李出现错误，失效")
                    out.put(i)

        while not out.empty():
            outIndex = out.get()
            self.trackersThird.pop(outIndex)
            self.trackersPositionThird.pop(outIndex)
        return img


