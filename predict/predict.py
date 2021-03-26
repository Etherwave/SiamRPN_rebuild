import os
import cv2
import torch
import numpy as np
from model.SiamRPN import SiamRPN
from tracker.mytracker import SiamRPNTracker
from data.myvideo import Video
from data.mysubdataset import SubDataSet
from paint.paint_fuc import *
from data.bbox_func import *
torch.set_num_threads(1)
snapshot_path = "./../save/video.pt"
video_path = 'E:/dataset/VOT/VOT2019/glove'
# dataset_path = ["E:/dataset/VOT/VOT2019"]
dataset_path = ["E:/dataset/tc"]

def get_init(video):
    cx, cy, w, h = video.gt[0]
    return [cx, cy, w, h]

def test_video(video, tracker):
    first_frame = True
    cv2.namedWindow("predict", cv2.WND_PROP_FULLSCREEN)
    for frame_path in video.frames_path:
        frame = cv2.imread(frame_path)
        if first_frame:
            try:
                # init_rect = cv2.selectROI("test", frame, False, False)
                center = get_init(video)
                tracker.init(frame, center)
            except:
                exit()
            first_frame = False
        else:
            outputs = tracker.track(frame)
            center = outputs['center']
            corner = center2corner(center)
            x1, y1, x2, y2 = corner
            paint_rectangle(frame, x1, y1, x2, y2)
            cv2.imshow("predict", frame)
            cv2.waitKey(1)
            # cv2.waitKey()

def test_dataset(dataset_path, tracker):
    shuffle = False
    dataset = SubDataSet(dataset_path, shuffle)
    for i in range(len(dataset.videos)):
        video = dataset.videos[i]
        test_video(video, tracker)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    model = SiamRPN()

    # load model
    model.load_state_dict(torch.load(snapshot_path, map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = SiamRPNTracker(model, device)
    test_dataset(dataset_path, tracker)

if __name__ == '__main__':
    main()
