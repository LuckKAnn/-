# coding: utf-8
import time
from multiprocessing import Process

import numpy as np
import os
import tensorflow as tf
import  cv2
from object_detection.utils import label_map_util


import ImageUtils
from siameseMatch import siamese_match_detection
# 模型位置及相关信息
MODEL_NAME = 'suitcase_detction'
PATH_TO_FROZEN_GRAPH = r'./resources/models/frozen_inference_graph.pb'
PATH_TO_LABELS = r'./resources/models/suitcase_label_map.pbtxt'
NUM_CLASSES = 1

# 创建通用工具
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
# 创建匹配查询类
siamese_match = siamese_match_detection()
"""
    识别进程: 通过由视频读取进程通过前期识别和跟踪，确定的行李，进行再次识别。
    识别之后，要么存入显示队列(监控一)，要么调用匹配查询类中方法，查询匹配信息，再进行返回。
"""
class DetectionProcess(Process):

    def __init__(self,listWaitToDetecte,listWaitToShow):
        super().__init__()
        self.listWaitToDetecte = listWaitToDetecte
        self.listWaitToShow = listWaitToShow


    def run(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                    if not self.listWaitToDetecte.empty():

                        waitToDetecteContent = self.listWaitToDetecte.get()
                        image_np = waitToDetecteContent[0]
                        numOfCamera  = waitToDetecteContent[1]
                        height,width,a = image_np.shape
                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                        # Each box represents a part of the image where a particular object was detected.
                        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                        scores = detection_graph.get_tensor_by_name('detection_scores:0')
                        classes = detection_graph.get_tensor_by_name('detection_classes:0')
                        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                        (boxes, scores, classes, num_detections) = sess.run(
                                [boxes, scores, classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})
                        s_boxes = boxes[scores > 0.50]
                        if s_boxes.size!= 0:
                            ymin = int(s_boxes[0][0] * height)  # ymin
                            xmin = int(s_boxes[0][1] * width)  # xmin
                            ymax = int(s_boxes[0][2] * height)  # ymax
                            xmax = int(s_boxes[0][3] * width)  # xmax
                            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 10)
                            part = image_np[ymin:ymax, xmin:xmax]


                            if numOfCamera == 1:
                                part = ImageUtils.gamma_trans(part, 0.7)
                                part = ImageUtils.resizeImg(part)
                                self.listWaitToShow.put((part, numOfCamera))
                            else:
                                if numOfCamera==2:
                                    part = ImageUtils.gamma_trans(part, 0.6)
                                    # part = ImageUtils.hisEqulColor(part)
                                    # part = ImageUtils.clasheImg(part)
                                else:
                                    part = ImageUtils.gamma_trans(part, 0.7)
                                part = ImageUtils.resizeImg(part)
                                partInfo = siamese_match.detecte(part,int(time.time()))
                                self.listWaitToShow.put((partInfo, numOfCamera))
                        else:
                            image_np = ImageUtils.gamma_trans(image_np, 0.7)
                            image_np = ImageUtils.resizeImg(image_np)
                            if numOfCamera == 1:
                                self.listWaitToShow.put((image_np, numOfCamera))
                            else:
                                partInfo = siamese_match.detecte(image_np,int(time.time()))
                                self.listWaitToShow.put((partInfo, numOfCamera))




