import numpy as np

'''
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

def calc_bounding_square_size(w, h):
    # context_amount = 0.5 这里的处理和siameseFC一致，算一个(w+2p)*(h+2p)=A^2
    context_amount = 0.5
    p_2 = (w + h)
    square_size = round(np.sqrt((w + context_amount * p_2) * (h + context_amount * p_2)))
    return square_size

def center2squarecenter(center):
    cx, cy, w, h = center
    square_size = calc_bounding_square_size(w, h)
    square_center = [cx, cy, square_size, square_size]
    return square_center

def center2squarecorner(center):
    square_center = center2squarecenter(center)
    square_corner = center2corner(square_center)
    return square_corner


