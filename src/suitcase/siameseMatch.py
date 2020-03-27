################################################################################################
#
#       2020.1.8
#       -----------
#       siameseNet 网络test
#       1. 图像增强 https://blog.csdn.net/weixin_40793406/article/details/84867143
#       https://www.pytorchtutorial.com/pytorch-one-shot-learning/#Contrastive_Loss_function
#
################################################################################################
import re
import sys
import threading
import time
from multiprocessing import Queue

import cv2
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
sys.path.append('..')
from operator import itemgetter

import PIL
from PIL import Image

from torchvision import transforms as tfs
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os

image_width = 200
image_height = 200


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 输入为 200 x 200
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            # 对每个 channel 按照概率设置为 0
            nn.Dropout2d(p=.2),
            # 输出为 4 * 200 * 200

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
            # 输出为 8 * 200 * 200

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            # 输出为 8 * 200 * 200
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * image_width * image_height, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = torch.tensor([margin])

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = ((1 - label) * torch.pow(euclidean_distance, 2)
                            + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)) / 2
        return loss_contrastive

class Color():
    def __init__(self):
            # region 辅助函数
    # RGB2XYZ空间的系数矩阵
        self.M = np.array([[0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227]])
#区域划分
    def bgr_mapping(self,img_val):
        # 将bgr颜色分成256/64个区间做映射
        return int(img_val/64)
# RGb的加权数值计算
    def calc_bgr_hist(self,im):
        # 缩放尺寸减小计算量
        scale=0.5
        width,height= int(im.size[0]*scale),int(im.size[1]*scale)
        im = im.resize((width, height), Image.ANTIALIAS)
        pix = im.load()
        hist1={}; hist2={}; hist3={}
        L_width,R_width=int(0.1*width),int(0.9*width)
        L_height,R_height=int(0.1*height),int(0.9*height)
        for x in range(L_width,R_width):
            for y in range(L_height,R_height):
                if(len(pix[x,y])==3):
                    maped_r, maped_g, maped_b= pix[x, y]
                else:
                    maped_r, maped_g, maped_b ,appra= pix[x, y]
                    if(appra==0):
                        continue
                # 计算像素值
                maped_b = self.bgr_mapping(maped_b)
                maped_g = self.bgr_mapping(maped_g)
                maped_r = self.bgr_mapping(maped_r)
                hist1[maped_b] = hist1.get(maped_b, 0) + 1
                hist2[maped_g] = hist2.get(maped_g, 0) + 1
                hist3[maped_r] = hist3.get(maped_r, 0) + 1
        B_L=sorted(hist1.items(),key=itemgetter(1),reverse=True)
        G_L=sorted(hist2.items(),key=itemgetter(1),reverse=True)
        R_L=sorted(hist3.items(),key=itemgetter(1),reverse=True)

        for i in range(len(B_L),3):
            B_L.append([0,0])
        B_mean = (B_L[0][0]*B_L[0][1]+B_L[1][0]*B_L[1][1]+B_L[2][0]*B_L[2][1])/(1.0*B_L[0][1]+B_L[1][1]+B_L[2][1])

        for i in range(len(G_L),3):
            G_L.append([0,0])
        G_mean = (G_L[0][0]*G_L[0][1]+G_L[1][0]*G_L[1][1]+G_L[2][0]*G_L[2][1])/(1.0*G_L[0][1]+G_L[1][1]+G_L[2][1])

        for i in range(len(R_L),3):
            R_L.append([0,0])
        R_mean = (R_L[0][0]*R_L[0][1]+R_L[1][0]*R_L[1][1]+R_L[2][0]*R_L[2][1])/(1.0*R_L[0][1]+R_L[1][1]+R_L[2][1])

        return [B_mean*64,G_mean*64,R_mean*64]
    # im_channel取值范围：[0,1]
    def f(self,im_channel):
        return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931
    def anti_f(self,im_channel):
        return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787
    # endregion
    # region RGB 转 Lab
    # 像素值RGB转XYZ空间，pixel格式:(B,G,R)
    # 返回XYZ空间下的值
    def __rgb2xyz__(self,pixel):
        b, g, r = pixel[0], pixel[1], pixel[2]
        rgb = np.array([r, g, b])
        # rgb = rgb / 255.0
        # RGB = np.array([gamma(c) for c in rgb])
        XYZ = np.dot(self.M, rgb.T)
        XYZ = XYZ / 255.0
        return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)
    def __xyz2lab__(self,xyz):
        """
        XYZ空间转Lab空间
        :param xyz: 像素xyz空间下的值
        :return: 返回Lab空间下的值
        """
        F_XYZ = [self.f(x) for x in xyz]
        L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
        a = 500 * (F_XYZ[0] - F_XYZ[1])
        b = 200 * (F_XYZ[1] - F_XYZ[2])
        return (L, a, b)


    def RGB2Lab(self,pixel):
        """
        RGB空间转Lab空间
        :param pixel: RGB空间像素值，格式：[G,B,R]
        :return: 返回Lab空间下的值
        """
        xyz = self.__rgb2xyz__(pixel)
        Lab = self.__xyz2lab__(xyz)
        return Lab


    # endregion

    # region Lab 转 RGB
    def __lab2xyz__(self,Lab):
        fY = (Lab[0] + 16.0) / 116.0
        fX = Lab[1] / 500.0 + fY
        fZ = fY - Lab[2] / 200.0

        x = self.anti_f(fX)
        y = self.anti_f(fY)
        z = self.anti_f(fZ)

        x = x * 0.95047
        y = y * 1.0
        z = z * 1.0883

        return (x, y, z)


    def __xyz2rgb(self,xyz):
        xyz = np.array(xyz)
        xyz = xyz * 255
        rgb = np.dot(np.linalg.inv(self.M), xyz.T)
        # rgb = rgb * 255
        rgb = np.uint8(np.clip(rgb, 0, 255))
        return rgb


    def Lab2RGB(self,Lab):
        xyz = self.__lab2xyz__(Lab)
        rgb = self.__xyz2rgb(xyz)
        return rgb
    # endregion

    #CIEDE
    def compare_similar_hist(self,rgb_1, rgb_2):
        lab_l,lab_a,lab_b=self.RGB2Lab(rgb_1)
        color1 = LabColor(lab_l, lab_a, lab_b)

        lab_l,lab_a,lab_b=self.RGB2Lab(rgb_2)
        color2 = LabColor(lab_l, lab_a, lab_b)
        delta_e = delta_e_cie2000(color1, color2)
        # delta_e = delta_e_cmc(color1, color2)
        return delta_e
    # 读取图片内容
    def color_test(self,srcpath,dstpath):
        # im1 = Image.open(srcpath)
        # im2 = Image.open(dstpath)
        im1 = srcpath
        im2 = dstpath
        end_ans=self.compare_similar_hist(self.calc_bgr_hist(im1), self.calc_bgr_hist(im2))
        return end_ans

class SiameseNetworkDataset():
    __set_size__ = 90
    __batch_size__ = 10

    def __init__(self, set_size=90, batch_size=10, transform=None, should_invert=False):
        self.imageFolderDataset = []
        self.train_dataloader = []

        self.__set_size__ = set_size
        self.__batch_size__ = batch_size

        self.transform = tfs.Compose([
            tfs.Resize((image_width, image_height)),
            tfs.ToTensor()
        ])
        self.should_invert = should_invert

    def __getitem__(self, class_num=40):
        '''
        如果图像来自同一个类，标签将为0，否则为1
        TODO: 实际上: Y值为1或0。如果模型预测输入是相似的，那么Y的值为0，否则Y为1。
        TODO: 由于classed_pack 每类可能有2-3张, 此时参数item_num无效,故删去参数中的item_num
        '''
        data0 = torch.empty(0, 3, image_width, image_height)
        data1 = torch.empty(0, 3, image_width, image_height)

        should_get_same_class = random.randint(0, 1)
        for i in range(self.__batch_size__):
            img0_class = random.randint(0, class_num - 1)
            # we need to make sure approx 50% of images are in the same class

            if should_get_same_class:
                item_num = len(self.imageFolderDataset[img0_class])
                temp = random.sample(list(range(0, item_num)), 2)
                img0_tuple = (self.imageFolderDataset[img0_class][temp[0]], img0_class)
                img1_tuple = (self.imageFolderDataset[img0_class][temp[1]], img0_class)
            else:
                img1_class = random.randint(0, class_num - 1)
                # 保证属于不同类别
                while img1_class == img0_class:
                    img1_class = random.randint(0, class_num - 1)
                item_num = len(self.imageFolderDataset[img0_class])
                img0_tuple = (self.imageFolderDataset[img0_class][random.randint(0, item_num - 1)], img0_class)
                item_num = len(self.imageFolderDataset[img1_class])
                img1_tuple = (self.imageFolderDataset[img1_class][random.randint(0, item_num - 1)], img1_class)

            img0 = Image.open(img0_tuple[0])
            img1 = Image.open(img1_tuple[0])

            if self.should_invert:
                # 二值图像黑白反转,默认不采用
                img0 = PIL.ImageOps.invert(img0)
                img1 = PIL.ImageOps.invert(img1)

            if self.transform is not None:
                img0 = self.transform(img0)
                img1 = self.transform(img1)

            img0 = torch.unsqueeze(img0, dim=0).float()
            img1 = torch.unsqueeze(img1, dim=0).float()

            data0 = torch.cat((data0, img0), dim=0)
            data1 = torch.cat((data1, img1), dim=0)

        # XXX: 注意should_get_same_class的值
        return data0, data1, torch.from_numpy(np.array([should_get_same_class ^ 1], dtype=np.float32))

    def classed_pack(self):
        # local = 'image/classed_pack/2019-03-14 22-19-img/'
        # local1 = 'image/classed_pack/2019-03-14 16-30-img/'
        local = 'image/classed_pack/2019-03-15 13-10-img/'
        local1 = 'image/classed_pack/2019-03-14 23-33-img/'
        self.imageFolderDataset = []

        # floader1
        subFloader = os.listdir(local)
        for i in subFloader:
            temp = []
            sub_dir = local + i + '/'
            subsubFloader = os.listdir(sub_dir)
            for j in subsubFloader:
                temp.append(sub_dir + j)
            self.imageFolderDataset.append(temp)
        # floader2
        subFloader = os.listdir(local1)
        for i in subFloader:
            temp = []
            sub_dir = local1 + i + '/'
            subsubFloader = os.listdir(sub_dir)
            for j in subsubFloader:
                temp.append(sub_dir + j)
            self.imageFolderDataset.append(temp)

        # 为数据集添加数据
        for i in range(self.__set_size__):
            img0, img1, label = self.__getitem__(len(self.imageFolderDataset))
            self.train_dataloader.append((img0, img1, label))


class siamese_match_detection():
    secondMatchQueue = Queue()
    thirdMatchQueue = Queue()

    def __init__(self):
        self.net = SiameseNetwork()
        self.net.load_state_dict(torch.load('./resources/models/net032201_normal_params.pkl'))
        self.net.eval()
        self.image_width = 200
        self.image_height = 200
        self.transform = tfs.Compose([
            tfs.Resize((self.image_width, self.image_height)),
            tfs.ToTensor()
        ])
        self.color = Color()

    def detecte(self,image,times):
        '''
        传入数据: 为numpy类型: [height, width, channel]
        返回数据: result(匹配为0, 不匹配为1), 误差(不匹配的误差大)
        '''
        src =  Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        self.dstpath = "./resources/img/first/"
        filelist = os.listdir(self.dstpath)
        length = 0
        list_len = len(filelist)
        resultImg = None
        resultPoint  = 100
        resultNo = None
        for i in filelist:
            fileTimeLine = int(re.findall("\d+",i)[0])
            if fileTimeLine<=times and fileTimeLine>times-100:
                dstimg = Image.open(self.dstpath + i)
                color_ans = self.color.color_test(src,dstimg)
                img0 = self.transform(src)
                img1 = self.transform(dstimg)
                img0 = torch.unsqueeze(img0, dim=0).float()
                img1 = torch.unsqueeze(img1, dim=0).float()
                data0 = torch.empty(0, 3, image_width, image_height)
                data1 = torch.empty(0, 3, image_width, image_height)
                data0 = torch.cat((data0, img0), dim=0)
                data1 = torch.cat((data1, img1), dim=0)
                # 调整维度顺序
                # img0 = img0.permute(0, 2, 3, 1)
                # img1 = img1.permute(0, 2, 3, 1)

                output1, output2 = self.net(data0, data1)
                euclidean_distance = F.pairwise_distance(output1, output2)
                sia_ans =  euclidean_distance.cpu().data.numpy()[0]
                # res = 0.7 * (color_ans / 8.0) + 0.3 * sia_ans
                res = 0.8 * (color_ans / 8.0) + 0.2 * sia_ans

            # TODO: 这两个参数是根据下面的调试结果定的
            # if euclidean_distance < 1.0:
                if res<resultPoint:
                    resultImg = dstimg
                    resultPoint = res
                    resultNo = i
                # return 0, euclidean_distance.cpu().data.numpy()[0]
            # elif euclidean_distance >= 1.0:
            #     return 1, euclidean_distance.cpu().data.numpy()[0]
        # print(resultPoint)
        # print(resultNo)
        if resultImg is None:
            print("没有匹配行李------------------")
        else:
            resultImg  =  cv2.cvtColor(np.asarray(resultImg),cv2.COLOR_RGB2BGR)
            show = (image, resultImg, resultNo.split(".")[0], resultPoint)
            return  show
            # if (times.split(":")[0] == "second"):
            #     siamese_match_detection.secondMatchQueue.put_nowait(show)
            #     # self.main.addimage(show, self.main.secondRow, 2)
            #     # self.main.secondRow += 1
            # else:
            #     siamese_match_detection.thirdMatchQueue.put_nowait(show)
            #     # self.main.addimage(show, self.main.thirdRow, 3)
            #     # self.main.thirdRow += 1

if __name__  =="__main__":
    img = cv2.imread(r"./resources/img/third/1.jpg")
    siamese_match_detection().detecte(img)