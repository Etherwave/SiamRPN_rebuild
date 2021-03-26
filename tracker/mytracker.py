import numpy as np
from data.bbox_func import *
from data.Anchor import MyAnchors
from data.Augmentation import crop_hwc
import cv2
import copy
import torch

class SiamRPNTracker(object):
    def __init__(self, model, device):
        super(SiamRPNTracker, self).__init__()
        self.device = device
        self.template_size = 127
        self.search_size = 255
        self.score_map_size = 17
        self.anchor_num = 5
        self.hanning_size = 17
        self.init_hanning_window()
        self.anchors_center = MyAnchors().anchors[1]
        self.model = model
        self.model.eval()

    def init_hanning_window(self):
        def build_hanning_window(hanning_size, score_map_size):
            '''
            制作一个hanning窗，用于抑制漂移，
            这里由于可能要试一试hanning窗大小大于score_size的情况，所以下面比较复杂
            :param hanning_size:
            :return:一个hanning窗
            '''
            hanning_size = int(hanning_size)
            if hanning_size < score_map_size:
                # 当hanning窗小于score_map_size时，要在边上补充
                # np.pad 就是在边缘填充
                pad = score_map_size - hanning_size
                d1 = int(pad / 2)
                d2 = pad - d1
                hanning = np.hanning(hanning_size)
                window = np.outer(hanning, hanning)
                window = np.pad(window, ((d1, d2), (d1, d2)), constant_values=((0, 0)))
            else:
                ds = hanning_size - score_map_size
                d1 = int(ds / 2)
                d2 = ds - d1
                hanning = np.hanning(hanning_size)
                window = np.outer(hanning, hanning)
                window = window[d1:hanning_size - d2].transpose()
                window = window[d1:hanning_size - d2].transpose()
            window = np.array([window for i in range(5)])
            return window
        self.window = build_hanning_window(self.hanning_size, self.score_map_size)

    def _convert_bbox(self, delta):
        '''
        将网络输出的delta即，[cx, cy, w, h]转为在search_size(255)这个尺度的一个值，
        之前做label的时候不是归一化，还有取log了嘛
        :param delta:
        :return:
        '''
        # delta第一维是batch_size,先去掉，去掉后为(20, 17, 17), 20指4*5
        delta = torch.squeeze(delta, dim=0)
        # 拆开
        delta = delta.contiguous().view(4, 5, 17, 17)
        delta = delta.data.cpu().numpy()
        predict_x, predict_y, predict_w, predict_h = delta

        anchor_x, anchor_y, anchor_w, anchor_h = self.anchors_center

        # 复原数据
        predict_x = predict_x*anchor_w+anchor_x
        predict_y = predict_y*anchor_h+anchor_y
        predict_w = np.exp(predict_w)*anchor_w
        predict_h = np.exp(predict_h)*anchor_h

        # 拼装返回
        predict = np.array([predict_x, predict_y, predict_w, predict_h])
        return predict

    def _convert_score(self, score):
        # 去掉batch_size这一维，剩下(5, 17, 17)
        score = torch.squeeze(score, dim=0)
        score = score.cpu().detach().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def crop_template(self, image):
        # 根据上一帧的位置和大小算出要裁剪的搜索区域
        crop_corner = center2squarecorner(self.center)
        # 裁剪搜索区域
        avg = np.mean(image, axis=(0, 1))
        image_crop = crop_hwc(image, crop_corner, self.template_size, padding=avg)
        # cv2.imshow("template", image_crop)
        # cv2.waitKey()
        # 图像要送入网络要从(w, h, c)->(c, w, h)，并且增加一个batch_size维度
        image_crop = np.transpose(image_crop, (2, 0, 1)).astype(np.float32)
        image_crop = image_crop[np.newaxis, :, :, :]
        image_crop = torch.from_numpy(image_crop).to(self.device)
        return image_crop

    def crop_search(self, image):
        # 根据目标上一帧的位置裁剪搜索图像
        square_center = center2squarecenter(self.center)
        cx, cy, w, h = square_center
        w = w / self.template_size * self.search_size
        h = h / self.template_size * self.search_size
        crop_corner = center2corner([cx, cy, w, h])
        avg = np.mean(image, axis=(0, 1))
        image_crop = crop_hwc(image, crop_corner, self.search_size, padding=avg)
        # cv2.imshow("search", image_crop)
        # cv2.waitKey()
        image_crop = np.transpose(image_crop, (2, 0, 1)).astype(np.float32)
        image_crop = image_crop[np.newaxis, :, :, :]
        image_crop = torch.from_numpy(image_crop).to(self.device)
        return image_crop

    def init(self, first_image, center):
        """
        根据第一张图片和gt计算好模板图像的future（特征图像）
        """
        self.center = center

        # crop
        z_crop = self.crop_template(first_image)

        self.model.template(z_crop)

    def track(self, image):
        x_crop = self.crop_search(image)

        # 正向推理一次
        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        predict = self._convert_bbox(outputs['loc'])
        predict_x, predict_y, predict_w, predict_h = predict

        # 计算等价正方形
        def calc_square_size(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # 加长宽比惩罚和面积惩罚
        def change(r):
            return np.maximum(r, 1. / r)

        def show_score_map(score, name):
            score_copy = copy.deepcopy(score)
            score_copy_min = np.min(score_copy)
            score_copy -= score_copy_min
            score_copy_max = np.max(score_copy)
            score_copy = score_copy / score_copy_max
            score_copy *= 255
            score_copy = score_copy.astype(np.uint8)
            for i in range(self.anchor_num):
                one_anchor_score = score_copy[i]
                score_pic = np.array([one_anchor_score, one_anchor_score, one_anchor_score])
                score_pic = np.transpose(score_pic, (1, 2, 0))
                score_pic = cv2.resize(score_pic, (300, 300))
                cv2.imshow(f"{name} score{i}", score_pic)
            cv2.waitKey()

        # show_score_map(score, "ori")

        # 面积惩罚
        cx, cy, w, h = self.center
        s_z = calc_square_size(w, h)
        # pred_bbox的长宽是相对于search_size(255)的裁剪图像的，我们上一帧存储的self.center里边是真实值，
        # 需要先用calc_bounding_square_size转到裁剪图像体系的比例(裁剪图像体系，就是指真实目标对应的square_size为127的体系)
        s_c = change(calc_square_size(predict_w, predict_h) /
                     (calc_square_size(w/s_z*self.template_size, h/s_z*self.template_size)))
        # 长宽比惩罚
        r_c = change((w/h) / (predict_w/predict_h))
        # 代码里边的惩罚是下面这种写法，与论文不一致
        k = 0.04
        # k = 10
        penalty = np.exp(-(r_c * s_c - 1) * k)
        pscore = penalty * score

        # show_score_map(pscore, "pscore")

        # window penalty
        # h = 0.2
        # h = 0.3
        # h = 0.44
        h = 0.6
        pscore = pscore * (1 - h) + self.window * h

        # show_score_map(pscore, "refine_score")

        # 找出最大的score的位置
        best_idx = np.unravel_index(np.argmax(pscore), pscore.shape)
        # predict是(4, 5, 17, 17)，我们从pscore找到的坐标是(5,17,17)的，所以加个:,去掉第一维
        best_center = predict[:, best_idx[0], best_idx[1], best_idx[2]]
        # 上边的4个值是将相对于search_size(255)的xywh值，按照比例放缩到真实大小
        # s_z是127长度对应的真实大小，那么255下的坐标转化方式应如下
        best_center = best_center * (s_z/self.template_size)
        best_x, best_y, best_w, best_h = best_center

        # 可知best_center预测的xy是相对于搜索图像的位置（搜索图像左上角就是坐标(0,0)的意思），
        # 搜索图像的中心是上一帧目标的中心self.center[0],self.center[1]，我们怎么算出现在的目标真正位置呢？
        # 利用目前预测的位置，相对与搜索图像中心的偏移来计算best_x-search_image_center_xy
        search_image_center_xy = (self.search_size/2)*(s_z/self.template_size)
        cx = self.center[0] + (best_x-search_image_center_xy)
        cy = self.center[1] + (best_y-search_image_center_xy)

        # 一个学习率，best_idx处的penalty越高，score越高，lr阅读，越相信预测值
        lr = penalty[best_idx] * score[best_idx] * 0.3

        # smooth bbox
        width = self.center[2] * (1 - lr) + best_w * lr
        height = self.center[3] * (1 - lr) + best_h * lr

        # clip boundary 防止预测出界
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, image.shape[:2])
        # update state
        self.center = [cx, cy, width, height]

        best_score = score[best_idx]
        return {
            'center': self.center,
            'best_score': best_score
        }
