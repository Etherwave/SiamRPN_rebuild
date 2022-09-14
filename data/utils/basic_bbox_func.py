import numpy as np
import math

'''
ltxywh 指左上角的ltx,lty，w，h
corner 指左上角一个点x1,y1,右下角一个点x2,y2
center 指中心一个点x,y,还有对应的w,h
'''

def corner2center(corner):
    x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
    x = (x1 + x2) * 0.5
    y = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h

def center2corner(center):
    x, y, w, h = center[0], center[1], center[2], center[3]
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return x1, y1, x2, y2

def ltxywh2corner(ltxywh):
    ltx, lty, w, h = ltxywh
    x1, y1, x2, y2 = ltx, lty, ltx+w, lty+h
    return [x1, y1, x2, y2]

def ltxywh2center(ltxywh):
    ltx, lty, w, h = ltxywh
    cx, cy, w, h = ltx+w/2, lty+h/2, w, h
    return [cx, cy, w, h]

def corner2ltxywh(corner):
    x1, y1, x2, y2 = corner
    ltx, lty, w, h = x1, y1, x2-x1, y2-y1
    return [ltx, lty, w, h]

def center2ltxywh(center):
    cx, cy, w, h = center
    ltx, lty, w, h = cx-w/2, cy-h/2, w, h
    return [ltx, lty, w, h]

def fourpoint2center(fourpoint):
    cx = np.mean(fourpoint[0::2])
    cy = np.mean(fourpoint[1::2])
    x1 = min(fourpoint[0::2])
    x2 = max(fourpoint[0::2])
    y1 = min(fourpoint[1::2])
    y2 = max(fourpoint[1::2])
    A1 = np.linalg.norm(fourpoint[0:2] - fourpoint[2:4]) * np.linalg.norm(fourpoint[2:4] - fourpoint[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    return cx, cy, w, h

def calc_iou(corner1, corner2):
    a_x1, a_y1, a_x2, a_y2 = corner1
    b_x1, b_y1, b_x2, b_y2 = corner2
    min_x = max(a_x1, b_x1)
    max_x = min(a_x2, b_x2)
    min_y = max(a_y1, b_y1)
    max_y = min(a_y2, b_y2)
    dx = max(0, max_x-min_x)
    dy = max(0, max_y-min_y)
    join_area = dx*dy
    a_area = (a_x2-a_x1)*(a_y2-a_y1)
    b_area = (b_x2-b_x1)*(b_y2-b_y1)
    iou = join_area/(a_area+b_area-join_area)
    return iou

def calc_ltxywh_iou(ltxywh1, ltxywh2):
    corner1 = ltxywh2corner(ltxywh1)
    corner2 = ltxywh2corner(ltxywh2)
    a_x1, a_y1, a_x2, a_y2 = corner1
    b_x1, b_y1, b_x2, b_y2 = corner2
    min_x = max(a_x1, b_x1)
    max_x = min(a_x2, b_x2)
    min_y = max(a_y1, b_y1)
    max_y = min(a_y2, b_y2)
    dx = max(0, max_x - min_x)
    dy = max(0, max_y - min_y)
    join_area = dx * dy
    a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    iou = join_area / (a_area + b_area - join_area)
    return iou

def calc_iou_np(corner1, corner2):
    # overlap
    a_x1, a_y1, a_x2, a_y2 = corner1[..., 0], corner1[..., 1], corner1[..., 2], corner1[..., 3]
    b_x1, b_y1, b_x2, b_y2 = corner2[..., 0], corner2[..., 1], corner2[..., 2], corner2[..., 3]

    min_x = np.maximum(a_x1, b_x1)
    max_x = np.minimum(a_x2, b_x2)
    min_y = np.maximum(a_y1, b_y1)
    max_y = np.minimum(a_y2, b_y2)

    w = np.maximum(0, max_x - min_x)
    h = np.maximum(0, max_y - min_y)

    a_area = (a_x2-a_x1) * (a_y2-a_y1)
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
    inter = w * h
    iou = inter / (a_area + b_area - inter)
    return iou