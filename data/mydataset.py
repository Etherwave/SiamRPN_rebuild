import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import math
from data.myvideo import Video
from data.mysubdataset import SubDataSet
import random
from data.Anchor import MyAnchors
from data.Augmentation import Augmentation
from data.bbox_func import corner2center, center2corner
from paint.paint_fuc import *
import copy

class MyDataSet(Dataset):
    def __init__(self):
        super(MyDataSet, self).__init__()
        self.anchors = MyAnchors()
        self.sample_random = random.Random()
        self.gray = 0.25
        self.neg = 0.2
        self.x_size = 255
        self.z_size = 127
        self.score_map_size = 17
        self.template_aug = Augmentation(shift=4/127, scale=0.05)
        self.search_aug = Augmentation(shift=64/127, scale=0.18, blur=0.18)
        self.croped_frame_paths = []
        self.max_video_frames = 0
        datasets = ["E:/dataset/VOT/VOT2019"]
        # datasets = ["E:/dataset/tc/"]
        self.dataset = SubDataSet(datasets)
        self.size = self.dataset.size

    def __len__(self):
        return self.size

    def random(self):
        return self.sample_random.random()

    def IoU(self, rect1, rect2):
        # overlap
        x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
        tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

        xx1 = np.maximum(tx1, x1)
        yy1 = np.maximum(ty1, y1)
        xx2 = np.minimum(tx2, x2)
        yy2 = np.minimum(ty2, y2)

        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)

        area = (x2 - x1) * (y2 - y1)

        target_a = (tx2 - tx1) * (ty2 - ty1)

        inter = ww * hh
        overlap = inter / (area + target_a - inter)

        return overlap

    def build_label(self, anchor, gt_corner, score_map_size, neg):
        self.thr_high = 0.6
        self.thr_low = 0.3
        self.negative_max_num = 16
        self.positive_max_num = 16
        self.total_max_num = 64

        anchor_num = anchor.anchors.shape[2]

        # cls 分类标签中-1表示负样本 0表示忽略 1表示正样本
        cls = np.zeros((anchor_num, score_map_size, score_map_size), dtype=np.int64)
        # delta指(x-anchor_cx)/anchor_w,(y-anchor_cy)/anchor_h,log(w/anchor_w),log(h/anchor_h)
        delta = np.zeros((4, anchor_num, score_map_size, score_map_size), dtype=np.float32)
        # delta_weight用来计算完逻辑回归损失更新时,每个anchor占的权重，可知只有正样本才有正确的逻辑回归，才需要权重
        delta_weight = np.zeros((anchor_num, score_map_size, score_map_size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        if neg:
            # 在整张图片中随机挑选一些位置设置成-1
            neg, neg_num = select(np.where(cls == 0), self.negative_max_num)
            cls[neg] = -1
            return cls, delta, delta_weight

        tcx, tcy, tw, th = corner2center(gt_corner)
        # anchor_box x1 x2 y1 y2 当初生成anchor的时候生成了两种格式，一种是corner，还有一种center
        anchor_box = anchor.anchors[0]
        # anchor_center cx cy w h
        anchor_center = anchor.anchors[1]
        # 对于anchor_box(4,5,25,25) 4->代表两个点坐标
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        # 对于anchor_center(4,5,25,25) 4->代表cxcywh
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]

        # tcx-cx指目标距离当前anchor中心位置的x之差，除以anchor的w，得到距离x比
        delta[0] = (tcx - cx) / w
        # tcy-cy指目标距离当前anchor中心位置的y之差，除以anchor的h，得到距离y比
        delta[1] = (tcy - cy) / h
        # 真实大小tw/预测大小w=大小比，预测大了的话>1小了<1,分别是正数和负数
        # 这里已经准备使用NLLOSE了，-sum(y*log(pre)),这里直接把结果取了log
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        # 查看delta的图像
        # delta_x = copy.deepcopy(delta[0])
        # for i in range(5):
        #     one_delta_x = delta_x[i]
        #     min_bias = np.min(one_delta_x)
        #     one_delta_x -= min_bias
        #     max_bias = np.max(one_delta_x)
        #     one_delta_x /= max_bias
        #     one_delta_x *= 255
        #     one_delta_x = np.array([one_delta_x, one_delta_x, one_delta_x]).transpose((1, 2, 0)).astype(np.uint8)
        #     cv2.imshow(f"delta_x{i}", one_delta_x)
        # delta_y = copy.deepcopy(delta[1])
        # cv2.waitKey()
        # for i in range(5):
        #     one_delta_y = delta_y[i]
        #     min_bias = np.min(one_delta_y)
        #     one_delta_y -= min_bias
        #     max_bias = np.max(one_delta_y)
        #     one_delta_y /= max_bias
        #     one_delta_y *= 255
        #     one_delta_y = np.array([one_delta_y, one_delta_y, one_delta_y]).transpose((1, 2, 0)).astype(np.uint8)
        #     cv2.imshow(f"delta_y{i}", one_delta_y)
        # cv2.waitKey()

        # IoU
        overlap = self.IoU([x1, y1, x2, y2], gt_corner)
        # 并交比大于0.6认为是正样本
        # 并交比小于0.3认为是负样本

        # 查看iou的图像
        # for i in range(5):
        #     one_overlap = overlap[i]
        #     one_overlap*=255
        #     one_overlap = np.array([one_overlap, one_overlap, one_overlap])
        #     one_overlap = np.transpose(one_overlap, (1, 2, 0)).astype(np.uint8)
        #     one_overlap = cv2.resize(one_overlap, (50, 50))
        #     cv2.imshow(f"overlap{i}", one_overlap)
        # cv2.waitKey()

        # 找出iou符合正负样本要求的位置
        pos = np.where(overlap > self.thr_high)
        neg = np.where(overlap < self.thr_low)

        # 选出一些正负样本
        pos, pos_num = select(pos, self.positive_max_num)
        neg, neg_num = select(neg, self.total_max_num-pos_num)

        # 正样本的位置我标记为了1
        cls[pos] = 1
        # 负样本的位置我标记为了-1
        cls[neg] = -1
        # 逻辑回归时使用的delta_weight,只有正样本时,搜索区域内有目标，才可以逻辑回归
        # 若存在一些anchor能够有较大的iou,那么pos_num>0,用这些anchor来回归位置，使他们平分权重
        # 若不存在一些anchor有与目标有较大的iou,那么pos_num=0,pos为空,下面这行代码相当于没有运行
        # 所以原作者代码中的1e-6是完全没有必要的,没有用的,这个式子永远不会出现除零错误的
        delta_weight[pos] = 1. / (pos_num + 1e-6)

        return cls, delta, delta_weight

    def __getitem__(self, index):
        # print(index)
        gray = self.gray and self.gray > self.random()
        neg = self.neg and self.neg > self.random()

        if neg:
            frame1, frame2, gt1, gt2 = self.dataset.get_negtive_pair()
        else:
            frame1, frame2, gt1, gt2 = self.dataset.get_positive_pair()

        z_image = cv2.imread(frame1)
        x_image = cv2.imread(frame2)

        z_center = gt1
        x_center = gt2

        # 进行数据增强，平移变换，放缩变换，裁剪出127*127的样板
        template, _gt_corner = self.template_aug(z_image, z_center, self.z_size, gray=gray)
        # 同样进行数据增强，裁剪出255*255的样本
        search, gt_corner = self.search_aug(x_image, x_center, self.x_size, gray=gray)

        # 关键的生成答案的部分
        cls, delta, delta_weight = self.build_label(self.anchors, gt_corner, self.score_map_size, neg)
        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])
        # template(3,127,127)search(3,255,255)cls(5,25,25)delta(4,5,25,25)delta_wight(5,25,25)

        return template, search, cls, delta, delta_weight, gt_corner


if __name__ == '__main__':
    md = MyDataSet()
    cnt = 20
    for i in range(cnt):
        template, search, cls, delta, delta_weight, gt_corner = md.__getitem__(i)
        template, search = map(lambda x: np.transpose(x, (1, 2, 0)).astype(np.uint8), [template, search])
        x1, y1, x2, y2 = gt_corner
        paint_rectangle(search, x1, y1, x2, y2)
        cv2.imshow("template", template)
        cv2.imshow("search", search)
        for i in range(5):
            one_cls = cls[i]
            cnt = 0
            for p in range(17):
                for q in range(17):
                    if one_cls[p][q]==-1:
                        cnt += 1
            # print(cnt)
            one_cls = one_cls+1
            one_cls *= 100
            one_cls = np.array([one_cls, one_cls, one_cls])
            one_cls = np.transpose(one_cls, (1, 2, 0)).astype(np.uint8)
            one_cls = cv2.resize(one_cls, (50, 50))
            cv2.imshow(f"{i}", one_cls)
        cv2.waitKey()

