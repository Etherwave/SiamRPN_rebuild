import os
from paint.paint_func import *
from data.utils.basic_bbox_func import *
from config.config import config

UAV123_path = "E:/dataset/Dataset-UAV-123/data/UAV123"

class Video:

    def __init__(self, video_path, gt_path, att_path, configSeqs_path):
        self.video_path = video_path
        self.video_name = self.video_path[self.video_path.rfind("/")+1:]
        self.gt_path = gt_path
        self.att_path = att_path
        self.configSeqs_path = configSeqs_path
        self.start_frame = -1
        self.end_frame = -1
        self.frames = []
        self.gt = []
        self.att = []
        self.get_data()

    def get_video_start_frame_end_frame(self):
        configSeqs_file = open(self.configSeqs_path, "r")
        lines = configSeqs_file.readlines()
        configSeqs_file.close()

        UAV20L_gt_name_start_frame_end_frame = {}
        UVA123_gt_name_start_frame_end_frame = {}
        UAV20L_flag = True
        for i in range(len(lines)):
            if len(lines[i]) <= 100:
                continue
            name_start = lines[i].find(",")+2
            if name_start != 0:
                name_end = lines[i].find("'", name_start+1)
                video_name = lines[i][name_start:name_end]
                start_frame_start_index = lines[i].find("startFrame")+12
                start_frame_end_index = lines[i].find(",", start_frame_start_index)
                start_frame = int(lines[i][start_frame_start_index:start_frame_end_index])

                end_frame_start_index = lines[i].find("endFrame")+10
                end_frame_end_index = lines[i].find(",", end_frame_start_index)
                end_frame = int(lines[i][end_frame_start_index:end_frame_end_index])
                # print(video_name, start_frame, end_frame)
                if UAV20L_flag:
                    UAV20L_gt_name_start_frame_end_frame[video_name] = {"start_frame": start_frame,
                                                                        "end_frame": end_frame}
                else:
                    UVA123_gt_name_start_frame_end_frame[video_name] = {"start_frame": start_frame,
                                                                        "end_frame": end_frame}
            if lines[i][-2] == ';':
                UAV20L_flag = False

        gt_name_start = self.gt_path.rfind("/")+1
        gt_name_end = self.gt_path.rfind(".")
        gt_name = self.gt_path[gt_name_start:gt_name_end]
        # print(UVA123_gt_name_start_frame_end_frame.keys())
        video_info = UVA123_gt_name_start_frame_end_frame[gt_name]
        return video_info["start_frame"], video_info["end_frame"]

    def get_data(self):
        # get gt
        self.gt = np.genfromtxt(self.gt_path, delimiter=",", dtype=int)

        # get frames path
        self.frames = []
        frame_names = [int(f[:-4]) for f in os.listdir(self.video_path)]
        frame_names.sort()
        for i in range(len(frame_names)):
            frame_path = self.video_path + "/%06d.jpg" % (frame_names[i])
            self.frames.append(frame_path)
        # print(len(self.frames))
        self.start_frame, self.end_frame = self.get_video_start_frame_end_frame()
        self.frames = self.frames[self.start_frame-1:self.end_frame]
        # print(len(self.frames))
        # print(len(self.gt))
        assert len(self.frames) == len(self.gt)
        # get att
        self.att = np.genfromtxt(self.att_path, delimiter=",", dtype=int)


class UAV123_DataSet:

    def __init__(self, UAV123_path):
        self.dataset_name = "UAV123"
        self.UAV123_path = UAV123_path
        self.UAV123_anno_path = self.UAV123_path+"/anno/UAV123"
        self.UVA123_video_path = self.UAV123_path+"/data_seq/UAV123"
        self.configSeqs_path = self.UAV123_path + "/configSeqs.m"
        self.videos = []
        self.get_data()

    def get_data(self):
        self.videos = []
        gt_names = [f for f in os.listdir(self.UAV123_anno_path) if f.endswith("txt")]
        '''
        UAV123是指用于短时跟踪的数据集，其中有一些视频的gt被分成了几部分，如bird1视频对应的gt有bird1_1，bird1_2，bird1_3
        并且其视频参数类型的标注也不一样，在短时跟踪是，应该把这一个视频分成多个小视频来对待
        '''
        for i in range(len(gt_names)):
            gt_path = self.UAV123_anno_path + "/" + gt_names[i]
            video_name = ""
            video_name_end_index = gt_names[i].find("_")
            if video_name_end_index == -1:
                video_name=gt_names[i][:-4]
            elif gt_names[i][video_name_end_index+1]=='s':
                video_name = gt_names[i][:-4]
            else:
                video_name = gt_names[i][:video_name_end_index]
            video_path = self.UVA123_video_path + "/" + video_name
            att_path = self.UAV123_anno_path+"/att/"+gt_names[i]
            # print(video_path, gt_path, att_path)
            self.videos.append(Video(video_path, gt_path, att_path, self.configSeqs_path))



if __name__ == '__main__':
    UAV123_dataset_path = config["windows"].datasets.UAV123_dataset_path
    UAV123_DataSet(UAV123_dataset_path)
    # video_path = "E:/dataset/Dataset-UAV-123/data/UAV123/data_seq/UAV123/bird1"
    # gt_path = "E:/dataset/Dataset-UAV-123/data/UAV123/anno/UAV123/bird1_2.txt"
    # att_path = "E:/dataset/Dataset-UAV-123/data/UAV123/anno/UAV123/att/bird1_2.txt"
    # configSeqs_path = "E:/dataset/Dataset-UAV-123/data/UAV123/configSeqs.m"
    # Video(video_path, gt_path, att_path, configSeqs_path)