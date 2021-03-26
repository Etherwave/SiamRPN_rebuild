import os
import numpy as np
import random
from paint.paint_fuc import paint_rectangle
from data.bbox_func import fourpoint2center, center2corner
import cv2


class Video:

    def __init__(self, video_path):
        '''
        获取视频文件夹地址
        获取该视频每一帧的路径
        获取该视频的gt
        存储视频的帧数
        初始化要给subdataset传送哪一帧的名叫"下一帧"的变量的序号
        初始化该视频是否每一帧都用过了的flag，used_out
        将帧打乱顺序，实现随机给subdata传帧
        :param video_path:
        '''
        self.video_path = video_path
        self.get_frames_path()
        self.get_gt()
        self.size = len(self.gt)
        self.next_no = 0
        self.used_out = False
        self.init_order()

    def get_frames_path(self):
        self.frames_path = [self.video_path+'/'+f for f in os.listdir(self.video_path) if f.endswith('jpg')]

    def fourpoint_to_corner(self, gt):
        '''
        将vot的4个坐标，转化为ltx,lty,w,h
        ltx表述left top x，lty表示left top y
        :return:
        '''
        new_gt = []
        for i in range(len(gt)):
            cx, cy, w, h = fourpoint2center(gt[i])
            new_gt.append([cx, cy, w, h])
        return new_gt

    def ltcxwh2center(self, gt):
        new_gt = []
        for i in range(len(gt)):
            cx, cy, w, h = gt[i]
            cx, cy = cx + w/2, cy + h/2
            new_gt.append([cx, cy, w, h])
        return new_gt

    def get_gt(self):
        gt_path = self.video_path+'/groundtruth.txt'
        self.gt = np.genfromtxt(gt_path, delimiter=',')
        # vot 数据集是8个数值，表示一个四边形的4个点。
        if len(self.gt[0]) == 8:
            self.gt = self.fourpoint_to_corner(self.gt)
        else:
            self.gt = self.ltcxwh2center(self.gt)

    def init_order(self):
        '''
        一共有size这么多帧，每次随机很麻烦，不如先随机好一个顺序，然后一帧一帧传就行
        :return:
        '''
        self.order = np.arange(self.size)
        np.random.shuffle(self.order)

    def normal_order(self):
        self.order = np.arange(self.size)

    def get_next_no(self):
        '''
        下一帧的序号，更新used_out
        :return:
        '''
        no = self.next_no
        self.next_no += 1
        if self.next_no >= self.size:
            self.next_no = 0
            self.used_out = True
        return no

    def get_tow_frame(self):
        '''
        在该视频中获取两帧
        :return:
        '''
        no1 = self.order[self.get_next_no()]
        no2 = self.order[self.get_next_no()]
        frame1_path = self.frames_path[no1]
        frame2_path = self.frames_path[no2]
        gt1 = self.gt[no1]
        gt2 = self.gt[no2]
        return frame1_path, frame2_path, gt1, gt2

    def get_one_frame(self):
        '''
        在该视频中获取一帧
        :return:
        '''
        no = self.order[self.get_next_no()]
        frame_path = self.frames_path[no]
        gt = self.gt[no]
        return frame_path, gt

if __name__ == '__main__':
    video = Video("E:/dataset/VOT/VOT2019/girl")
    while video.used_out == False:
        print(f"{video.next_no}/{video.size}")
        frame_path, gt = video.get_one_frame()
        image = cv2.imread(frame_path)
        x1, y1, x2, y2 = center2corner(gt)
        paint_rectangle(image, x1, y1, x2, y2)
        cv2.imshow("1", image)
        cv2.waitKey()
