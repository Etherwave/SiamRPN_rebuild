import os
import cv2
import torch
import numpy as np
import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join
from data.mysubdataset import SubDataSet
from paint.paint_func import *
from data.bbox_func import *
from Evaluation.eval_standard import Eval_Standard

from model.SiamRPN import SiamRPN
from tracker.mytracker import SiamRPNTracker
import time

torch.set_num_threads(1)

model_path = '../save/video.pt'

dataset_path = ["E:/dataset/VOT/VOT2019"]
# dataset_path = ["E:/dataset/tc"]

eval_output_folder = "./eval_output/"

def eval_video(video, tracker, eval_info):
    cv2.namedWindow("evaluation", cv2.WND_PROP_FULLSCREEN)
    state = None
    need_reset = True
    wait_number = 0
    burn_in_period = 0

    for i in range(video.size):
        frame = cv2.imread(video.frames_path[i])
        if need_reset and wait_number <= 0:
            center = video.gt[i]
            tracker.init(frame, center)

            if i > 0:
                eval_info.failure_cnt += 1
            burn_in_period = eval_info.burn_in_period-1
            need_reset = False

            center = tracker.center
            x1, y1, x2, y2 = center2corner(center)
            paint_rectangle(frame, x1, y1, x2, y2)
            cv2.imshow("evaluation", frame)
            cv2.waitKey(1)
            # cv2.waitKey()

        else:
            tracker.track(frame)  # track
            center = tracker.center
            corner = center2corner(center)
            gt_corner = center2corner(video.gt[i])
            iou = calc_iou(gt_corner, corner)

            if wait_number > 0:
                wait_number -= 1
            else:
                if iou > eval_info.failure_iou_threshold:
                    if burn_in_period > 0:
                        burn_in_period -= 1
                    else:
                        eval_info.total_iou += iou
                        eval_info.success_track_frames_cnt += 1
                else:
                    need_reset = True
                    wait_number = eval_info.wait_number-1

            x1, y1, x2, y2 = corner
            paint_rectangle(frame, x1, y1, x2, y2)
            cv2.imshow("evaluation", frame)
            cv2.waitKey(1)
            # cv2.waitKey()

def eval_dataset(dataset_path, tracker):
    eval_info = Eval_Standard()
    shuffle = False
    dataset = SubDataSet(dataset_path, shuffle)
    video_number = len(dataset.videos)
    # video_number = 2
    for i in range(video_number):
        video = dataset.videos[i]
        eval_video(video, tracker, eval_info)
    try:
        eval_info.average_iou = eval_info.total_iou/eval_info.success_track_frames_cnt
    except:
        eval_info.average_iou = 0
    eval_info.robustness = eval_info.failure_cnt/video_number
    print(f"average iou is: {eval_info.average_iou}")
    print(f"robustness is: {eval_info.robustness}")
    if os.path.exists(eval_output_folder) == False:
        os.mkdir(eval_output_folder)

    t = time.localtime()
    year, month, day, h, m, s = t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec
    name = f"{year}_{month}_{day}_{h}_{m}_{s}"
    output_file_path = eval_output_folder+name+".txt"
    output_file = open(output_file_path, "w")
    output_file.write(f"average iou is: {eval_info.average_iou}\n")
    output_file.write(f"robustness is: {eval_info.robustness}\n")
    for i in range(len(dataset.dataset_path)):
        output_file.write(dataset.dataset_path[i]+"\n")
    output_file.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    model = SiamRPN()

    # load model
    model.load_state_dict(torch.load(join(realpath(dirname(__file__)), model_path)))
    model.eval().to(device)

    # build tracker
    tracker = SiamRPNTracker(model, device)
    eval_dataset(dataset_path, tracker)

if __name__ == '__main__':
    main()
