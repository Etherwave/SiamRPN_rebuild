import os
import torch
from data.Normal_Video_Dataset.DTB70_Datset import DTB70_DataSet
from data.Normal_Video_Dataset.TC_Dataset import TC_DataSet
from data.Normal_Video_Dataset.UAV123_Dataset import UAV123_DataSet
from model.SiamRPN import SiamRPN
from tracker.SiamRPNTracker import SiamRPNTracker
from data.utils.SiamRPN.bbox_func import *
from paint.paint_func import *
import cv2
from config.config import config

system_type = "windows" if os.name == "nt" else "linux"

save_path = config[system_type].project_path + "/save"

model_path = save_path+"/SiamRPN_epoch_12.pt"

burn_in = 5

test_one_video_result_folder = config[system_type].project_path+"/test/SiamRPN/test_one_video_result"

def mkdir(path):
    if os.path.exists(path) == False:
        os.mkdir(path)

def test_one_video(dataset_name, video_name):
    mkdir(test_one_video_result_folder)
    # load dataset
    if dataset_name == "TC":
        dataset = TC_DataSet(config[system_type].datasets.TC_dataset_path)
    elif dataset_name == "DTB70":
        dataset = DTB70_DataSet(config[system_type].datasets.DTB70_dataset_path)
    elif dataset_name == "UAV123":
        dataset = UAV123_DataSet(config[system_type].datasets.UAV123_dataset_path)
    else:
        print(dataset_name + " not exists!")
        return

    cuda_no = config[system_type].cuda_no
    device = torch.device("cuda", cuda_no)

    # build model
    model = SiamRPN(device)

    # load model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    # build tracker
    tracker = SiamRPNTracker(model, device)

    show = True

    video = None
    for i in range(len(dataset.videos)):
        if dataset.videos[i].video_name == video_name:
            video = dataset.videos[i]
            break
    if video==None:
        print(video_name+" not exist!")
        return

    pred_bboxes = []
    j = 0
    init_frame = True
    lost_number = 0
    while j < len(video.frames):
        image = cv2.imread(video.frames[j])
        gt = video.gt[j]

        if init_frame:
            init_frame = False
            if gt[2]<1 or gt[3]<1:
                init_frame = True
                pred_bboxes.append(0)
            else:
                tracker.init(image, gt)
                pred_bboxes.append(1)
            if show:
                corner = ltxywh2corner(gt)
                x1, y1, x2, y2 = corner
                paint_text(image, 5, 15, str(j+1), color=colors[2])
                paint_text(image, 5, 35, str(lost_number), color=colors[2])
                paint_rectangle(image, x1, y1, x2, y2, color=colors[2])
                cv2.imshow("test", image)
                # cv2.imwrite(test_one_video_result_folder + "/%04d.jpg" % j, image)
                cv2.waitKey(1)
            j += 1
        else:
            outputs = tracker.track(image)
            ltxywh = outputs['ltxywh']
            iou = calc_ltxywh_iou(gt, ltxywh)
            # print(gt)
            # print(ltxywh)
            # print(iou)
            if show:
                corner = ltxywh2corner(gt)
                x1, y1, x2, y2 = corner
                paint_text(image, 5, 15, str(j + 1), color=colors[2])
                paint_text(image, 5, 35, str(lost_number), color=colors[2])
                paint_rectangle(image, x1, y1, x2, y2, color=colors[2])
                pred_corner = ltxywh2corner(ltxywh)
                pred_x1, pred_y1, pred_x2, pred_y2 = pred_corner
                paint_rectangle(image, pred_x1, pred_y1, pred_x2, pred_y2, color=colors[1])
                cv2.imshow("test", image)
                # cv2.imwrite(test_one_video_result_folder+"/%04d.jpg" % j, image)
                cv2.waitKey(1)
            if iou > 0:
                # not lost
                pred_bboxes.append(ltxywh)
                j += 1
            else:
                # lost object 一旦丢失目标，frame_counter会比idx快5帧，那么接下来的5帧都会pred_bboxes.append(0)
                init_frame = True
                lost_number += 1
                pred_bboxes.append(2)
                k = 1
                j += 1
                while k < burn_in and j < len(video.frames):
                    k += 1
                    j += 1
                    pred_bboxes.append(0)

    result_file_path = test_one_video_result_folder+"/result.txt"
    result_file = open(result_file_path, "w", encoding='utf-8')
    for i in range(len(pred_bboxes)):
        if isinstance(pred_bboxes[i], int):
            result_file.write(str(pred_bboxes[i]))
        else:
            for j in range(len(pred_bboxes[i])):
                if j != 0:
                    result_file.write(",")
                result_file.write(str(pred_bboxes[i][j]))
        result_file.write("\n")
    result_file.close()


if __name__ == '__main__':
    dataset_name = "DTB70"
    video_name = "Car4"
    # dataset_name = "UAV123"
    # video_name = "bike1"
    test_one_video(dataset_name, video_name)