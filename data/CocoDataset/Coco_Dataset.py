import os
from paint.paint_func import *
from data.utils.basic_bbox_func import *
import json
import cv2

Coco_Dataset_path = "E:/dataset/coco"

refine_simple_gt_path = "E:/dataset/coco/gt/refine_simple_coco2017.json"

class Coco_DataSet:

    def __init__(self, Coco_Dataset_path, gt_path):
        self.Coco_Dataset_path = Coco_Dataset_path
        self.gt_path = gt_path
        self.image_path = self.Coco_Dataset_path+"/train2017"
        self.annotations = []
        self.categories = []
        self.get_data()

    def get_data(self):
        self.annotations = []
        self.categories = []

        gt_file = open(self.gt_path, "r", encoding='utf-8')
        gt_json = json.load(gt_file)
        gt_file.close()

        self.annotations = gt_json["annotations"]
        self.categories = gt_json["categories"]

def show():
    dataset = Coco_DataSet(Coco_Dataset_path, refine_simple_gt_path)

    for i in range(len(dataset.annotations)):
        print(dataset.categories[i])
        if len(dataset.annotations[i]) == 0:
            continue
        one_category_annotations = dataset.annotations[i]
        for j in range(len(one_category_annotations)):
            annotation = one_category_annotations[j]
            image_id, bbox = annotation
            image_path = dataset.image_path+"/%012d.jpg" % (image_id)
            image = cv2.imread(image_path)
            x1, y1, x2, y2 = ltxywh2corner(bbox)
            paint_rectangle(image, x1, y1, x2, y2)
            cv2.imshow("1", image)
            cv2.waitKey()

if __name__ == '__main__':
    show()