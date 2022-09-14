import os
from paint.paint_func import *
from data.utils.basic_bbox_func import *

DTB70_path = "E:/dataset/DTB70/DTB70"

class Video:

    def __init__(self, video_path):
        self.video_path = video_path
        video_name_start = self.video_path.rfind("/")+1
        self.video_name = self.video_path[video_name_start:]
        self.frames = []
        self.gt = []
        self.get_data()

    def get_data(self):
        # get frames path
        self.frames = []
        img_path = self.video_path+"/img"
        frame_names = [int(f[:-4]) for f in os.listdir(img_path)]
        frame_names.sort()
        for i in range(len(frame_names)):
            frame_path = img_path+"/%05d.jpg" % (frame_names[i])
            self.frames.append(frame_path)
        # get gt
        self.gt = []
        self.gt = np.genfromtxt(self.video_path+"/groundtruth_rect.txt", delimiter=",", dtype=int)



class DTB70_DataSet:

    def __init__(self, DTB70_path="E:/dataset/DTB70/DTB70"):
        self.dataset_name = "DTB70"
        self.DTB70_path = DTB70_path
        self.videos = []
        self.get_data()

    def get_data(self):
        self.videos = []
        video_names = os.listdir(self.DTB70_path)
        for i in range(len(video_names)):
            video_path = self.DTB70_path+"/"+video_names[i]
            self.videos.append(Video(video_path))


def show():
    dataset = DTB70_DataSet(DTB70_path)
    i = 0
    videos = dataset.videos
    play_stop = False
    play_one_frame = False
    while i < len(videos):
        print(videos[i].video_path)
        print(videos[i].video_name)
        frames = videos[i].frames
        gt = videos[i].gt
        j = 0
        while j < len(frames):
            if play_stop == False or play_one_frame == True:
                play_one_frame = False
                image_path = frames[j]
                image = cv2.imread(image_path)
                paint_text(image, 5, 15, str(j+1), color=colors[0])
                ltxywh = gt[j]
                ltx, lty, w, h = ltxywh
                if w >= 0 and h >= 0:
                    corner = ltxywh2corner(ltxywh)
                    x1, y1, x2, y2 = corner
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    paint_rectangle(image, x1, y1, x2, y2, color=colors[0 % len(colors)])
                cv2.imshow("1", image)
                j += 1
            # cv2.waitKey()
            c = cv2.waitKey(1)
            c = c & 0xFF
            if c == ord('q'):
                exit(0)
            elif c == ord(' '):
                play_stop = not play_stop
            elif c == ord('d'):
                play_one_frame = True
                break
            elif c == ord('a'):
                i -= 2
                i = max(-1, i)
                play_one_frame = True
                break
            elif c == ord('z'):
                j -= 2
                j = max(0, j)
                play_one_frame = True
            elif c == ord('x'):
                play_one_frame = True

        i += 1

if __name__ == '__main__':
    show()