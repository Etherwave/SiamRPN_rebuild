import os
import torch
from data.Normal_Dataset.TC_Dataset import TC_DataSet
from data.Normal_Dataset.DTB70_Datset import DTB70_DataSet
from data.Normal_Dataset.UAV123_Dataset import UAV123_DataSet
from model.SiamRPNPP import SiamRPNPP
from tracker.SiamRPNPPTracker import SiamRPNPPTracker
from data.utils.SiamRPNPP.bbox_func import *
from paint.paint_func import *
import cv2
from config.config import config
import time

system_type = "windows" if os.name == "nt" else "linux"

result_save_folder = config[system_type].project_path + "/test/results"

save_path = config[system_type].project_path + "/save"

# model_path = save_path+"/SiamRPNPP_epoch_220.pt"
# model_path = save_path+"/SiamRPNPP_video_8800.pt"
model_path = save_path+"/SiamRPNPP_epoch_5_coco.pt"

algorithm_name = "SiamRPNPP"

burn_in = 5

def mkdir(path):
    if os.path.exists(path) == False:
        os.mkdir(path)

def test():
    # load dataset
    # dataset_names = ["DTB70", "UAV123"]
    dataset_names = ["DTB70"]
    # dataset_names = ["UAV123"]
    for dataset_name in dataset_names:
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
        device = torch.device('cuda', cuda_no)

        # build model
        model = SiamRPNPP(device)

        # load model
        # model.load_state_dict(torch.load(model_path))
        print(model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.cuda().eval()

        # build tracker
        tracker = SiamRPNPPTracker(model, device)

        mkdir(result_save_folder)
        result_algorithm_folder = result_save_folder+"/"+algorithm_name
        mkdir(result_algorithm_folder)
        result_algorithm_dataset_folder = result_algorithm_folder+"/"+dataset.dataset_name
        mkdir(result_algorithm_dataset_folder)
        current_test_folder = result_algorithm_dataset_folder
        mkdir(current_test_folder)

        # show = True
        show = False

        start_time = time.time()

        i = 0
        while i < len(dataset.videos):
            video = dataset.videos[i]
            print(video.video_name, str(i+1)+"/"+str(len(dataset.videos)))
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
                        cv2.waitKey(1)
                    j += 1
                else:
                    outputs = tracker.track(image)
                    ltxywh = outputs['ltxywh']
                    iou = calc_ltxywh_iou(gt, ltxywh)
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
            video_result_save_path = current_test_folder + "/" + video.video_name + ".txt"
            file = open(video_result_save_path, "w", encoding="utf-8")
            for j in range(len(pred_bboxes)):
                if isinstance(pred_bboxes[j], int):
                    file.write(str(pred_bboxes[j]))
                else:
                    for k in range(len(pred_bboxes[j])):
                        if k != 0:
                            file.write(",")
                        file.write(str(pred_bboxes[j][k]))
                file.write("\n")
            file.close()
            i += 1

        end_time = time.time()
        cost_time = end_time-start_time

        print("cost time %.2fs" % (cost_time))

        cost_time_save_path = current_test_folder+"/time.txt"
        file = open(cost_time_save_path, "w", encoding="utf-8")
        file.write(str(cost_time))
        file.close()



if __name__ == '__main__':
    test()