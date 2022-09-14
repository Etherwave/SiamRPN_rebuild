import numpy as np
import torch
import torch.nn.functional as F

def calc_loc_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)

def calc_cls_loss(cls, label_cls, device):
    b = cls.size()[0]
    cls = cls.view(-1)
    label_cls = label_cls.view(-1)
    pos = label_cls.data.eq(1).nonzero(as_tuple=False).squeeze().to(device)
    neg = label_cls.data.eq(-1).nonzero(as_tuple=False).squeeze().to(device)
    pos_loss = torch.pow((torch.index_select(cls, 0, pos)-1), 2)
    neg_loss = torch.pow((torch.index_select(cls, 0, neg)), 2)
    cls_loss = (pos_loss.sum()+neg_loss.sum())/b
    return cls_loss

def calc_loss(cls, loc, label_cls, label_loc, label_loc_weight, device):
    cls_loss = calc_cls_loss(cls, label_cls, device)
    loc_loss = calc_loc_loss(loc, label_loc, label_loc_weight)
    return cls_loss, loc_loss

