import torch.nn as nn
import torch.nn.functional as F
import torch
from model.Alexnet import AlexNet
from loss.SiamRPNPP_Origin_cross_entropy_loss.loss import select_cross_entropy_loss, weight_l1_loss

def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class DepthwiseRPN(nn.Module):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc



class SiamRPN(nn.Module):
    def __init__(self, device):
        super(SiamRPN, self).__init__()
        self.device = device
        # build backbone
        self.backbone = AlexNet()
        # build rpn head
        self.rpn_head = DepthwiseRPN()

    def template(self, z):
        zf = self.backbone(z)
        self.zf = zf

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def track(self, x):
        xf = self.backbone(x)
        cls, loc = self.rpn_head(self.zf, xf)

        return {
                'cls': cls,
                'loc': loc,
               }

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].to(self.device)
        search = data['search'].to(self.device)
        label_cls = data['label_cls'].to(self.device)
        label_loc = data['label_loc'].to(self.device)
        label_loc_weight = data['label_loc_weight'].to(self.device)

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        # print(zf.shape)
        # print(xf.shape)
        # torch.Size([1, 256, 6, 6])
        # torch.Size([1, 256, 22, 22])

        cls, loc = self.rpn_head(zf, xf)

        # print(cls.shape)
        # print(loc.shape)
        # torch.Size([1, 5, 17, 17])
        # torch.Size([1, 20, 17, 17])

        # cls = torch.sigmoid(cls)
        cls = self.log_softmax(cls)

        # get loss
        cls_loss = select_cross_entropy_loss(cls, label_cls, self.device)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
        # # get loss
        # cls_loss, loc_loss = calc_loss(cls, loc, label_cls, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = 1.0 * cls_loss + 1.2 * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        return outputs

def calc_model_size(model):
    parameter_number = sum([param.nelement() for param in model.parameters()])
    mb_size = parameter_number*4/1024/1024
    return mb_size


if __name__ == '__main__':
    device = torch.device("cpu")
    model = SiamRPN(device)
    # print(model)
    print("Number of parameter: %.2fMB" % (calc_model_size(model)))