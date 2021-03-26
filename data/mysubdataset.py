import os
import numpy as np
import random
from paint.paint_fuc import paint_rectangle
import cv2
from data.myvideo import Video
from data.bbox_func import center2corner

class SubDataSet():

    def __init__(self, dataset_path, shuffle=True):
        '''
        获取数据集文件夹路径
        创建存放所有视频的list
        保存该数据集帧数的总量
        获取所有视频
        :param dataset_path:
        '''
        self.dataset_path = dataset_path
        self.videos = []
        self.size = 0
        self.shuffle = shuffle
        self.get_videos()

    def get_videos(self):
        '''
        获取本数据集的所有视频
        并打乱视频顺序
        :return:
        '''
        videos_path = []
        for dataset_path in self.dataset_path:
            videos_path.extend(
                [dataset_path+'/'+f for f in os.listdir(dataset_path)
                            if os.path.isdir(dataset_path+'/'+f)]
            )

        for path in videos_path:
            video = Video(path)
            self.size += video.size
            self.videos.append(video)

        if self.shuffle:
            self.shuffle_video()

    def reset_videos_used_state(self):
        '''
        重置所有视频的使用状态
        :return:
        '''
        for video in self.videos:
            video.used_out = False
            video.next_no = 0

    def shuffle_video(self):
        '''
        打乱视频的顺序，保证了每次训练使用的视频顺序不同
        :return:
        '''
        np.random.shuffle(self.videos)

    def get_positive_pair(self):
        '''
        获取一对正样本
        :return:
        '''
        frame1 = None
        frame2 = None
        gt1 = None
        gt2 = None
        # 视频的顺序已经打乱过了，直接按顺序访问就行
        for video in self.videos:
            if video.used_out == False:
                frame1, frame2, gt1, gt2 = video.get_tow_frame()
                break
        # 如果该数据集所有数据都已经使用过了，但是还是向该数据集请求获取正样本，那么就重置状态，重新打乱，并获取一对正样本
        if frame1==None:
            self.reset_videos_used_state()
            self.shuffle_video()
            frame1, frame2, gt1, gt2 = self.get_positive_pair()
        return frame1, frame2, gt1, gt2

    def get_negtive_pair(self):
        frame1 = None
        frame2 = None
        gt1 = None
        gt2 = None
        get_first = True
        # 用get_first 标志来表示现在读取到第几个图片了
        for video in self.videos:
            if video.used_out == False:
                if get_first:
                    frame1, gt1 = video.get_one_frame()
                    get_first = False
                else:
                    frame2, gt2 = video.get_one_frame()
                    break
        # 如果有任何一个为None，表面数据集不够用了，重置一遍flag，就是设置几百个bool值，也快，重新打乱，然后获取
        if frame1 == None or frame2 == None:
            self.reset_videos_used_state()
            self.shuffle_video()
            frame1, frame2, gt1, gt2 = self.get_negtive_pair()
        return frame1, frame2, gt1, gt2

if __name__ == '__main__':
    datasets_path = ["E:/dataset/VOT/VOT2019"]
    subd = SubDataSet(datasets_path)

    def test_positive():
        f1_path, f2_path, gt1, gt2 = subd.get_positive_pair()
        f1 = cv2.imread(f1_path)
        f2 = cv2.imread(f2_path)

        x1, y1, x2, y2 = center2corner(gt1)
        paint_rectangle(f1, x1, y1, x2, y2)

        x1, y1, x2, y2 = center2corner(gt2)
        paint_rectangle(f2, x1, y1, x2, y2)

        cv2.imshow("1", f1)
        cv2.imshow("2", f2)
        cv2.waitKey()

    def test_negative():
        f1_path, f2_path, gt1, gt2 = subd.get_negtive_pair()

        f1 = cv2.imread(f1_path)
        f2 = cv2.imread(f2_path)

        x1, y1, x2, y2 = center2corner(gt1)
        paint_rectangle(f1, x1, y1, x2, y2)

        x1, y1, x2, y2 = center2corner(gt2)
        paint_rectangle(f2, x1, y1, x2, y2)

        cv2.imshow("1", f1)
        cv2.imshow("2", f2)
        cv2.waitKey()

    cnt = 10

    for i in range(cnt):
        test_positive()

    for i in range(cnt):
        test_negative()

