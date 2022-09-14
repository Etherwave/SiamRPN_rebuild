# -*- coding: UTF-8 -*-

import math
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from data.Dataloader.SiamRPN.Dataset_For_Dataloader import Dataset_For_Dataloader
from model.SiamRPN import SiamRPN
from config.config import config

def get_time_str():
    t = time.localtime()
    return "%d_%d_%d_%d_%d_%d" % (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

system_type = "windows" if os.name == "nt" else "linux"

save_folder = config[system_type].train.model_save_folder
train_log_folder = config[system_type].project_path+"/train/SiamRPN_Train/train_log"

train_log_path = train_log_folder+"/"+get_time_str()+".txt"

# pretrain_model_path = save_folder + "/model.pth"
pretrain_model_path = save_folder + "/SiamRPN_epoch_12.pt"

epoch_save_model_path = save_folder + "/SiamRPN_epoch_{}.pt"
video_save_model_path = save_folder + "/SiamRPN_video.pt"

def server_print(info):
    info = str(info)
    print(info)
    file = open(train_log_path, "a", encoding='utf-8')
    file.write(info+"\n")
    file.close()

def mkdir(path):
    if os.path.exists(path)==False:
        os.mkdir(path)

def calc_time(start_time, end_time):
    s = int(end_time - start_time)
    m = int(s / 60)
    h = int(m / 60)
    s %= 60
    m %= 60
    return h, m, s


def get_max_epoch_number():
    models_path = [save_folder+"/"+f for f in os.listdir(save_folder)]
    max_no = 0
    no_start = epoch_save_model_path.rfind("_")
    epoch_save_model_path_prefix = epoch_save_model_path[:no_start]
    for i in range(len(models_path)):
        if models_path[i].startswith(epoch_save_model_path_prefix):
            no = models_path[i][len(epoch_save_model_path_prefix)+1:-3]
            try:
                no = int(no)
            except:
                no = 0
            max_no = max(max_no, no)
    return max_no


def train(model, device, optimizer, scheduler, train_loader):
    model = model.train()

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    loss_sum = 0
    batch_cnt = 0
    for i, data in enumerate(train_loader):
        data = {
            'template':         data[0].to(device),
            'search':           data[1].to(device),
            'label_cls':        data[2].to(device),
            'label_loc':        data[3].to(device),
            'label_loc_weight': data[4].to(device),
            'gt_cornor':        data[5],
        }

        outputs = model(data)
        loss = outputs['total_loss']
        server_print(loss.data.item())

        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += loss.data.item()
        batch_cnt += 1
    torch.save(model.state_dict(), video_save_model_path)
    server_print("epoch avg loss is {0}".format(loss_sum / batch_cnt))


def main():
    if os.path.exists(save_folder)==False:
        os.mkdir(save_folder)
    mkdir(train_log_folder)

    cuda_no = config[system_type].cuda_no
    device = torch.device('cuda', cuda_no)
    server_print(device)
    model = SiamRPN(device)
    # 加载模型写在to(device之前)

    if os.path.exists(pretrain_model_path):
        server_print("load model success!")
        model.load_state_dict(torch.load(pretrain_model_path, map_location=device))
        # model.load_state_dict(torch.load(pretrain_model_path, map_location={'cuda:0': 'cuda:2'}))
        server_print(pretrain_model_path)
    else:
        server_print("load model failed!")
        server_print(pretrain_model_path)
    # if os.path.exists(video_save_model_path):
    #     server_print("load model success!")
    #     model.load_state_dict(torch.load(video_save_model_path))
    #     server_print(video_save_model_path)
    # else:
    #     server_print("load model failed!")
    #     server_print(video_save_model_path)
    model = model.to(device)
    epoch = config[system_type].train.epoch

    # for p in model.backbone.parameters():
    #     p.requires_grad = False
    #
    # for p in model.neck.parameters():
    #     p.requires_grad = False

    initial_lr = config[system_type].train.lr
    # optimizer = torch.optim.AdamW(model.rpn_head.parameters(), lr=initial_lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))

    score_map_size = 17
    train_set = Dataset_For_Dataloader(score_map_size)
    batch_size = config[system_type].train.batch_size
    num_workers = config[system_type].train.num_workers

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

    total_epoch_start_time = time.time()
    server_print("start")
    for i in range(epoch):
        one_epoch_start_time = time.time()
        server_print("第%d个epoch的学习率：%.9f" % (i+1, optimizer.param_groups[0]['lr']))
        train(model, device, optimizer, scheduler, train_loader)
        scheduler.step()
        torch.save(model.state_dict(), epoch_save_model_path.format(get_max_epoch_number()+1))
        one_epoch_end_time = time.time()
        h, m, s = calc_time(one_epoch_start_time, one_epoch_end_time)
        server_print("epoch{0} 花费时间{1}h{2}m{3}s".format(i, h, m, s))
    total_epoch_end_time = time.time()
    h, m, s = calc_time(total_epoch_start_time, total_epoch_end_time)
    server_print("训练{0}轮， 共花费{1}h{2}m{3}s".format(epoch, h, m, s))


if __name__ == '__main__':
    main()
