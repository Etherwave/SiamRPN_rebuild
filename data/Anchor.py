import math
import numpy as np
from data.bbox_func import *
from paint.paint_fuc import *
import copy

class MyAnchors():
    def __init__(self):
        self.ratios = [0.33, 0.5, 1, 2, 3]
        self.anchors = []
        self.search_size = 255
        self.score_map_size = 17
        self.stride = 8
        self.anchor_num = len(self.ratios)
        self.init_anchor()

    def init_anchor(self):
        # 把这些不同大小的anchor放到search_map上，一共放score_map_size*score_map_size,即17*17个
        # 原作者不是在整个图像上平均放的，而是比较靠近中间，这样每个anchor的感受野基本都在search_map中，不会出去
        # 计算平均以8为步长，放在255的中心
        # 计算一个block的面积
        # 原作者的代码是先stride*stride=8*8=64，然后又将边长扩大8倍，
        # 也没有其他说明，所以这里直接用64*64当作一个anchor的面积
        area_size = 64*64
        w = []
        h = []
        for i in range(len(self.ratios)):
            # w/h = ratios, w*h=area_size -> h*ratios*h = area_size
            _h = math.sqrt(area_size/self.ratios[i])
            _w = area_size/_h
            _h = int(_h)
            _w = int(_w)
            w.append(_w)
            h.append(_h)
        # 5个anchor的w
        w = np.array(w)
        # 5个anchor的h
        h = np.array(h)
        # [0, 1*stride, 2*strid, 3*stride, ..., 16*stride], 指17个anchor的间距
        disp = np.arange(0, self.score_map_size)*self.stride
        # 为了把anchor放在255 * 255 搜索图像的中间位置，那么对应的第一个左上角anchor的位置偏移
        # （17-1）=16个，17个anchor中间有16个空隙，每个长度为stride，17个anchor占用大小为(17-1)*self.stride
        # 那么下面的start_bias即为第一个anchor距离边缘的距离
        start_bias = (255-(17-1)*self.stride)//2
        # cx为每个anchor的中心位置的x, cy为y
        cx = disp + start_bias
        cy = disp + start_bias
        # 由于我们构造出来的形状为一个(17,)的向量，我们要搞成一个(17, 17)的图的形状，所以先搞一个(5, 17, 17)
        # 5是指5个anchor
        zero = np.zeros((self.anchor_num, self.score_map_size, self.score_map_size), dtype=np.float32)
        # 5种anchor的cx, cy相同，所以重复5次，构造出来, reshape是为了与上面的zero相加，得到我们期望的结果
        cx = np.array([cx for i in range(5)]).reshape((5, 1, 17))
        cy = np.array([cy for i in range(5)]).reshape((5, 17, 1))
        w = w.reshape((5, 1, 1))
        h = h.reshape((5, 1, 1))
        # 将cx,cy,w,h都填到(17, 17)图像的对应位置，这里的函数解释见：
        # https://nothingishere.top/blog/read/JZ2W24DZ/NVQXAIDMMFRG2ZDBEDTJ5BHJQCQONFNQ465YI%3D%3D%3D/
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.anchors = np.array([np.stack([x1, y1, x2, y2]), np.stack([cx, cy, w, h])])

if __name__ == '__main__':
    a = MyAnchors()
    print(a.anchors.shape)
    # 查看所有的anchor摆放位置是否正确
    # 取出cx,cy,w,h
    cx, cy, w, h = a.anchors[1]
    # 由于5中anchor的cx,cy,w,h相同，我们任取一种
    cx, cy = cx[0], cy[0]
    image = np.uint8(np.zeros((255, 255, 3)) + 255)
    for i in range(17):
        for j in range(17):
            paint_point(image, int(cx[i][j]), int(cy[i][j]), point_size=3)
    cv2.imshow("1", image)
    cv2.waitKey()
    # 查看第一行的五种anchor是否摆放正确
    # 取出5种anchor的x1, y1, x2, y2
    x1, y1, x2, y2 = a.anchors[0]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i in range(17):
        _image = copy.deepcopy(image)
        for j in range(5):
            _x1, _y1, _x2, _y2 = x1[j][0][i], y1[j][0][i], x2[j][0][i], y2[j][0][i]
            paint_rectangle(_image, _x1, _y1, _x2, _y2, thickness=3, color=colors[j])
        cv2.imshow("1", _image)
        cv2.waitKey()
    # 查看第一列的五种anchor是否摆放正确
    for i in range(17):
        _image = copy.deepcopy(image)
        for j in range(5):
            _x1, _y1, _x2, _y2 = x1[j][i][0], y1[j][i][0], x2[j][i][0], y2[j][i][0]
            paint_rectangle(_image, _x1, _y1, _x2, _y2, thickness=3, color=colors[j])
        cv2.imshow("1", _image)
        cv2.waitKey()
