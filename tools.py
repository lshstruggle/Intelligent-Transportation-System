# tools.py

import cv2
import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage

class Colors:
    def __init__(self):
        pass

    def __call__(self, class_id, bright=False):
        # 你可以根据 class_id 返回不同颜色
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # 示例颜色
        return colors[class_id % len(colors)]

def img_cvread(file_path):
    # 读取图像并处理
    return cv2.imread(file_path)

def drawRectBox(img, location, class_name, font, color):
    # 绘制矩形框
    xmin, ymin, xmax, ymax = location
    img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
    img = cv2.putText(img, class_name, (int(xmin), int(ymin) - 10), font, 0.5, color, 2)
    return img



def cvimg_to_qpiximg(cv_img):
    # OpenCV 转 Qt 图片
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width

    # 如果是BGR格式（OpenCV默认），需要转换为RGB格式
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    # 创建 QImage 对象
    q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    # 返回 QPixmap 对象
    return QPixmap.fromImage(q_img)
