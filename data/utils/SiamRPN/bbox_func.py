import numpy as np
import math
from data.utils.basic_bbox_func import *

def calc_bounding_square_size(w, h):
    # context_amount = 0.5 这里的处理和siameseFC一致，算一个(w+0.5*2p)*(h+0.5*2p)=A^2
    context_amount = 0.5
    p_2 = (w + h)
    square_size = np.round(np.sqrt((w + context_amount * p_2) * (h + context_amount * p_2)))
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
