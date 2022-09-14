import random
from data.utils.SiamRPN.bbox_func import *
from paint.paint_func import *


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    '''
    裁剪放缩图像
    :param image:需要裁剪放缩的图像
    :param bbox:要裁剪的位置
    :param out_sz:要裁剪成的大小，由于这里我们都是要裁剪成正方形，所以只传一个数值就好
    :param padding:裁剪到图片之外的时候填充的颜色，这里传过来的是平均值
    :return:返回一个裁剪好的图片
    '''
    # 先将bbox转为了float,好像没啥用
    bbox = [float(x) for x in bbox]
    # a是指要输出的宽度是目前裁剪区域的宽度大小的多少倍
    # b是高的比例
    '''
    cv2.warpAffine 的仿射矩阵mapping实现的功能是：
    new_xy = mapping*old_xy
    下面的仿射矩阵实现的转换是
    new_x = a*x+c
    new_y = b*y+d
    具体函数解释见：
    '''
    # bbox[2]-bbox[0] 就是x2-x1就是宽度，例如200-100，要裁剪的区域是100个像素，但是直接减是99,这里区别于原作者代码使用加一
    a = out_sz / (bbox[2]-bbox[0]+1)
    b = out_sz / (bbox[3]-bbox[1]+1)
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


class Augmentation:
    def __init__(self, shift=4/127, scale=1, blur=False):
        # default args
        self.shift = shift
        self.scale = scale
        self.blur = blur
        # 创建一个随机数产生器
        self.sample_random = random.Random()
        # 用于光照变化
        self.rgbVar = np.array([
            [-0.55919361,  0.98062831, - 0.41940627],
            [1.72091413,  0.19879334, - 1.82968581],
            [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)

    def random(self):
        # random.random()随机产生一个-1-1之间的实数 *2 ->0-2 -1->-1-1
        return self.sample_random.random() * 2 - 1.0

    def blur_image_aug(self, image):
        def rand_kernel():
            size = np.random.randn(1)
            size = int(np.round(size)) * 2 + 1
            if size < 0: return None
            if random.random() < 0.5: return None
            size = min(size, 45)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel

        kernel = rand_kernel()

        if kernel is not None:
            image = cv2.filter2D(image, -1, kernel)
        return image

    def to_gray_aug(self, image):
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.zeros((grayed.shape[0], grayed.shape[1], 3), np.uint8)
        image[:, :, 0] = image[:, :, 1] = image[:, :, 2] = grayed
        return image

    def change_color_aug(self, image):
        # 这里引入了一个光照变化？ 随机bgr减去一个数字，颜色会发生变化
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        return image

    def shift_and_scale_aug(self, image, gt_corner, size):
        '''
        该函数实现对原图进行裁剪和图像增强
        :param image:原图
        :param gt_corner:gt的corner表示[x1, y1, x2, y2]
        :param size:要裁剪成为的大小
        :return:返回裁剪好的图像image和在该图像中的gt_corner
        '''

        gt_center = corner2center(gt_corner)
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_corner
        image_h, image_w = image.shape[:2]

        # paint_rectangle(image, gt_x1, gt_y1, gt_x2, gt_y2)
        # cv2.imshow("image", image)
        # cv2.waitKey()

        # 计算出对应gt的一个近似的正方形bbox,
        # 为了一会儿的裁剪放缩不影响图像中的长宽比，所以直接裁剪合适大小的正方形，然后等比例放缩
        square_center = center2squarecenter(gt_center)
        # 将gt转成的正方形的边长看作127份,根据size(127或255)放缩得出一个要裁剪的bbox
        cx, cy, w, h = square_center
        # 上边的w,h是根据gt算出来的一个正方形刚好框住目标的一个大小，我们把他当作模板图像的大小127
        # 放缩到我们需要的size大小(127或255，对于要裁剪模板图像就不需要放缩了，对于搜索图像将这个w按照127份，算出新的255份的w)
        object_size = 127
        w = w*1.0 / object_size * size
        h = h*1.0 / object_size * size
        # 得到要裁剪的位置
        crop_bbox_center = [cx, cy, w, h]

        # 随机放缩要裁剪的区域（图像增强）
        # self.random会随机出-1~1的值，所以这个放缩变化还是比较小的
        # self.scale 对于template是4/127 对于 search 是64/255
        scale_x, scale_y = ((1.0 + self.random() * self.scale), (1.0 + self.random() * self.scale))
        # 下面这个放缩比例控制会导致如果 w 比较大，image_w/w会把本应裁剪的大w，通过scale_x编程了接近image_w的大小，就是把image_w当作了w的上限
        # 这个上限对于template来讲是没有问题的，template理应更加准确的只包含目标
        # 但是对于search可能会导致，当原本目标占据原图的1/2以上时，截取的search的w会超过image_w，但是又会被这个上限压缩回来，
        # 导致w无法达到template是127份，而search是255份的效果，而是类似于template是127份而search是180份的效果。
        # 放置的anchor的大小是64x64，127/180*255=180，会导致目标在搜图图像中的大小过大，64*64大小的anchor产生的iou无法超越阈值0.6
        # 从而导致正样本缺失，应该让目标在search中的大小接近64*64，而目标在template中的大小接近127*127
        scale_x = min(scale_x, float(image_w) / w)
        scale_y = min(scale_y, float(image_h) / h)
        # 裁剪的区域按比例放大缩小
        cx, cy, w, h = crop_bbox_center
        w = w * scale_x
        h = h * scale_y
        crop_bbox_center = [cx, cy, w, h]
        crop_bbox_corner = center2corner(crop_bbox_center)

        # _x1, _y1, _x2, _y2 = crop_bbox_corner
        # paint_rectangle(image, _x1, _y1, _x2, _y2)
        # cv2.imshow("image", image)
        # cv2.waitKey()

        # 随机平移要裁剪的区域
        gt_square_size = calc_bounding_square_size(gt_center[2], gt_center[3])
        tx, ty = (self.random() * self.shift*gt_square_size, self.random() * self.shift*gt_square_size)
        x1, y1, x2, y2 = crop_bbox_corner
        # 保证平移不出去(很重要，平移过大搜索框中就可能没有目标了，没法制作label)
        left_max_shift = x2-gt_x2
        right_max_shift = gt_x1-x1
        up_max_shift = y2-gt_y2
        down_max_shift = gt_y1-y1

        tx = max(-left_max_shift, min(right_max_shift, tx))
        ty = max(-up_max_shift, min(down_max_shift, ty))
        # 裁剪区域平移
        crop_bbox_corner = [x1 + tx, y1 + ty, x2 + tx, y2 + ty]

        # _x1, _y1, _x2, _y2 = crop_bbox_corner
        # paint_rectangle(image, _x1, _y1, _x2, _y2)
        # cv2.imshow("image", image)
        # cv2.waitKey()

        # 用x1,y1记录了要裁剪的区域的左上角的那个点
        x1 = crop_bbox_corner[0]
        y1 = crop_bbox_corner[1]

        # 我们用的那个裁剪函数crop_hwc本质上是将图像平移，然后裁剪，再放缩
        # 计算裁剪后图像的gt
        # 裁剪后裁剪的左上角的坐标从(x1,y1)->(0,0)
        # 那么对应的gt的[gt_x1,gt_y1,gt_x2,gt_y2]->[gt_x1-x1,gt_y1-y1,gt_x2-x2,gt_y2-y2]
        shifted_gt_corner = [gt_corner[0] - x1, gt_corner[1] - y1, gt_corner[2] - x1, gt_corner[3] - y1]
        # 计算裁剪后放缩到对应大小的gt
        # 设本来的裁剪好的大小为(w,h),放缩到了(size,size)
        # 那么gt应从原来的(x1,y1,x2,y2)->(x1/w*size,y1/h*size,x2/w*size,y2/h*size)
        shifted_scaled_gt_corner = [shifted_gt_corner[0]/w*size, shifted_gt_corner[1]/h*size,
                                    shifted_gt_corner[2]/w*size, shifted_gt_corner[3]/h*size]
        # 计算平均值,填充裁剪到图像外形成的空白区域
        avg = np.mean(image, axis=(0, 1))
        # 裁剪并放缩成size大小
        image = crop_hwc(image, crop_bbox_corner, size, padding=avg)

        # cv2.imshow("image", image)
        # cv2.waitKey()

        return image, shifted_scaled_gt_corner

    def __call__(self, image, gt_corner, size, gray=False):
        '''
        此函数用来裁剪目标图像和搜索图像，不但有裁剪的功能，还包含了数据增强功能，
        包括：转化为灰度图，平移，放缩，颜色变化，图像模糊
        :param image: 需要裁剪的原图
        :param gt_corner: 目标的一个gt,[x1,y1,x2,y2]
        :param size:要裁剪成的大小，对于模板图像来说是127，对于搜索图像是255
        :param gray:是否要转化为灰度图
        :return:返回裁剪好的图像和目标在该裁剪好的图像的位置是一个corner,[x1,y1,x2,y2]
        '''

        # 平移放缩
        image, final_gt_corner = self.shift_and_scale_aug(image, gt_corner, size)

        # 如果是要变成灰度图的话，改成灰度图
        if gray:
            image = self.to_gray_aug(image)

        # cv2.imshow("image", image)
        # cv2.waitKey()

        # 颜色变化 这个变化后image的数值会变成float，不能直接显示
        image = self.change_color_aug(image)


        # cv2.imshow("image", np.array(image, dtype=np.uint8))
        # cv2.imshow("image", image)
        # cv2.waitKey()

        # 图像模糊
        if self.blur > random.random():
            image = self.blur_image_aug(image)

        return image, final_gt_corner
