import cv2
import numpy as np

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 255, 255)]

def paint_rectangle(frame, x1, y1, x2, y2, thickness=2, color=(0, 0, 255)):
    # 绘制一个红色矩形
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    ptLeftTop = (x1, y1)
    ptRightBottom = (x2, y2)
    thickness = thickness
    lineType = 8
    cv2.rectangle(frame, ptLeftTop, ptRightBottom, color, thickness, lineType)

def paint_text(frame, x, y, text, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
    frame = cv2.putText(frame, text, (x, y), font, 0.6, color, 2)

def paint_point(frame, x, y, color=(0, 0, 255), point_size=8):
    cv2.circle(frame, (x, y), point_size, color, -1)

def conver_score_to_color(score):
    max_num = score.max()
    score=score/max_num
    score*=255
    score = np.array(score, dtype=np.uint8)
    return score
