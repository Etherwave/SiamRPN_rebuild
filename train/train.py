import math
import os
import time
import torch
from torch.utils.data import DataLoader
from data.mydataset import MyDataSet
from model.SiamRPN import SiamRPN

save_path = "./../save/"
pretrain_model_path = save_path + "video.pt"
epoch_save_model_path = save_path + "epoch_{}.pt"
video_save_model_path = save_path + "video.pt"

def calc_time(start_time, end_time):
    s = int(end_time - start_time)
    m = int(s / 60)
    h = int(m / 60)
    s %= 60
    m %= 60
    return h, m, s

def get_max_epoch_number():
    epochs = os.listdir(save_path)
    max_no = 0
    for i in range(len(epochs)):
        if epochs[i][:5] == "epoch":
            no = epochs[i][6:-3]
            try:
                no = int(no)
            except:
                no = 0
            max_no = max(max_no, no)
    return max_no

def train(model, device):
    model = model.train()
    lr = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    train_set = MyDataSet()
    batch_size = 8
    num_worker = 4
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_worker,
                              pin_memory=True, sampler=None)
    video_loss = 0
    for i, data in enumerate(train_loader):
        data = {
            'template':         data[0].to(device),
            'search':           data[1].to(device),
            'label_cls':        data[2].to(device),
            'label_loc':        data[3].to(device),
            'label_loc_weight': data[4].to(device),
        }

        outputs = model(data)
        loss = outputs['total_loss']

        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        video_loss += loss.item()
        if i % 200 == 0 and i != 0:
            torch.save(model.state_dict(), video_save_model_path)
            print("avg loss is {0}".format(video_loss / ((i+1)*batch_size)))
    torch.save(model.state_dict(), video_save_model_path)
    print("epoch avg loss is {0}".format(video_loss / train_set.size))


def main():
    if os.path.exists(save_path)==False:
        os.mkdir(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SiamRPN()
    # 加载模型写在to(device之前)
    if os.path.exists(pretrain_model_path):
        model.load_state_dict(torch.load(pretrain_model_path))
        print("load model success!")
    model = model.to(device)
    epoch = 1
    totle_epoch_start_time = time.time()
    print("start")
    for i in range(epoch):
        one_epoch_start_time = time.time()
        train(model, device)
        torch.save(model.state_dict(), epoch_save_model_path.format(get_max_epoch_number()+1))
        one_epoch_end_time = time.time()
        h, m, s = calc_time(one_epoch_start_time, one_epoch_end_time)
        print("epoch{0} 花费时间{1}h{2}m{3}s".format(i, h, m, s))
    totle_epoch_end_time = time.time()
    h, m, s = calc_time(totle_epoch_start_time, totle_epoch_end_time)
    print("训练{0}轮， 共花费{1}h{2}m{3}s".format(epoch, h, m, s))


if __name__ == '__main__':
    main()

