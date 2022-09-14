import torch
import torch.nn.functional as F

def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    # print(pred.shape)
    # print(label.shape)
    # torch.Size([12, 2])
    # torch.Size([12])
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label, device):
    # print(pred.shape)
    # print(label.shape)
    # torch.Size([1, 5, 25, 25, 2])
    # torch.Size([1, 5, 25, 25])
    pred = pred.view(-1, 2)
    label = label.view(-1)
    # when use mydataset, i generate gt 1 for pos -1 for neg, but in the pysot 1 for pos 0 for neg.
    # pos = label.data.eq(1).nonzero().squeeze().cuda()
    # neg = label.data.eq(0).nonzero().squeeze().cuda()
    pos = label.data.eq(1).nonzero(as_tuple=False).squeeze().to(device)
    neg = label.data.eq(-1).nonzero(as_tuple=False).squeeze().to(device)
    # print(pos.shape)
    # print(neg.shape)
    # torch.Size([12])
    # torch.Size([12])
    # print(pos)
    # print(neg)
    # tensor([2030, 2031, 2032, 2055, 2056, 2606, 2630, 2631, 2655, 2656, 2680, 2681,
    #         2705, 2706], device='cuda:0')
    # tensor([3, 24, 723, 968, 1401, 2111, 2191, 2522, 3056, 3120],
    #        device='cuda:0')
    label[neg] = 0
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


