# -*- coding: utf-8 -*-
import time
from ultralytics import YOLO
import os
import cv2
import numpy as np
import tools
from PyQt5.QtWidgets import QFileDialog
from PIL import ImageFont
from PyQt5.QtWidgets import QApplication
import video
import camera
from PyQt5.QtWidgets import QInputDialog

app = QApplication([])

# 加载检测模型
model = YOLO('训练好的模型地址', task='detect')
model(np.zeros((48, 48, 3)))  # 预先加载推理模型



# 用于绘制不同颜色矩形框
colors = tools.Colors()

class Config:
    # 类别名称字典，键为类别编号，值为类别名称
    CH_names = {0: "tiny-car",1: "mid-car", 2: "big-car",3:"small-truck",4:"big-truck",5:"oil-truck",6:"special-car"}
    # 类别名称列表
    names = ['tiny-car', 'mid-car','big-car', 'small-truck', 'big-truck', 'oil-truck', 'special-car']
    # 保存路径
    save_path = "./output"

def open_img(file_path):
    if not file_path:
        return

    org_img = tools.img_cvread(file_path)

    # 目标检测
    t1 = time.time()
    results = model(file_path)[0]
    t2 = time.time()
    take_time_str = '{:.3f} s'.format(t2 - t1)
    print(take_time_str)

    location_list = results.boxes.xyxy.tolist()
    cls_list = results.boxes.cls.tolist()
    conf_list = results.boxes.conf.tolist()
    conf_list = ['%.2f %%' % (each * 100) for each in conf_list]

    total_nums = len(location_list)
    cls_percents = []
    for i in range(2):
        res = cls_list.count(i) / total_nums
        cls_percents.append(res)
    set_percent(cls_percents)

    now_img = results.plot()
    draw_img = now_img

    # 在图片左上角显示车辆总数
    cv2.putText(draw_img, f"Total Vehicles: {total_nums}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 保存识别后的图片
    save_detect_image(file_path, draw_img)


    # 设置路径显示
    print(f"Path: {file_path}")

    # 目标数目
    target_nums = len(cls_list)
    print(f"Target numbers: {target_nums}")

    if target_nums >= 1:
        print(f"Type: {Config.CH_names[cls_list[0]]}")
        print(f"Confidence: {conf_list[0]}")
        print(f"Xmin: {location_list[0][0]}")
        print(f"Ymin: {location_list[0][1]}")
        print(f"Xmax: {location_list[0][2]}")
        print(f"Ymax: {location_list[0][3]}")
    else:
        print("Type: ")
        print("Confidence: ")
        print("Xmin: ")
        print("Ymin: ")
        print("Xmax: ")
        print("Ymax: ")

    # 删除表格所有行
    tabel_info_show(location_list, cls_list, conf_list, path=file_path)

def detact_batch_imgs(directory):
    if not directory:
        return

    img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
    for file_name in os.listdir(directory):
        full_path = os.path.join(directory, file_name)
        if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
            img_path = full_path
            org_img = tools.img_cvread(img_path)

            # 目标检测
            t1 = time.time()
            results = model(img_path)[0]
            t2 = time.time()
            take_time_str = '{:.3f} s'.format(t2 - t1)
            print(take_time_str)

            location_list = results.boxes.xyxy.tolist()
            cls_list = results.boxes.cls.tolist()
            conf_list = results.boxes.conf.tolist()
            conf_list = ['%.2f %%' % (each * 100) for each in conf_list]

            total_nums = len(location_list)
            cls_percents = []
            for i in range(2):
                if total_nums > 0:
                   res = cls_list.count(i) / total_nums
                else:
                    res = 0
                cls_percents.append(res)
            set_percent(cls_percents)

            now_img = results.plot()

            # 在图片左上角显示车辆总数
            cv2.putText(now_img, f"Total Vehicles: {total_nums}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 保存识别后的图片
            save_detect_image(img_path, now_img)

            # 设置路径显示
            print(f"Path: {img_path}")

            # 目标数目
            target_nums = len(cls_list)
            print(f"Target numbers: {target_nums}")

            if target_nums >= 1:
                print(f"Type: {Config.CH_names[cls_list[0]]}")
                print(f"Confidence: {conf_list[0]}")
                print(f"Xmin: {location_list[0][0]}")
                print(f"Ymin: {location_list[0][1]}")
                print(f"Xmax: {location_list[0][2]}")
                print(f"Ymax: {location_list[0][3]}")
            else:
                print("Type: ")
                print("Confidence: ")
                print("Xmin: ")
                print("Ymin: ")
                print("Xmax: ")
                print("Ymax: ")

            # 删除表格所有行
            tabel_info_show(location_list, cls_list, conf_list, path=img_path)
            print("Table updated")

def draw_rect_and_tabel(results, img):
    now_img = img.copy()
    location_list = results.boxes.xyxy.tolist()
    cls_list = results.boxes.cls.tolist()
    conf_list = results.boxes.conf.tolist()
    conf_list = ['%.2f %%' % (each * 100) for each in conf_list]

    for loacation, type_id, conf in zip(location_list, cls_list, conf_list):
        type_id = int(type_id)
        color = colors(int(type_id), True)
        now_img = tools.drawRectBox(now_img, loacation, Config.CH_names[type_id], fontC, color)

    # 在图片左上角显示车辆总数
    total_nums = len(location_list)
    cv2.putText(now_img, f"Total Vehicles: {total_nums}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 保存识别后的图片
    save_detect_image(org_path, now_img)

    # 设置路径显示
    print(f"Path: {org_path}")

    # 目标数目
    target_nums = len(cls_list)
    print(f"Target numbers: {target_nums}")

    if target_nums >= 1:
        print(f"Type: {Config.CH_names[cls_list[0]]}")
        print(f"Confidence: {conf_list[0]}")
        print(f"Xmin: {location_list[0][0]}")
        print(f"Ymin: {location_list[0][1]}")
        print(f"Xmax: {location_list[0][2]}")
        print(f"Ymax: {location_list[0][3]}")
    else:
        print("Type: ")
        print("Confidence: ")
        print("Xmin: ")
        print("Ymin: ")
        print("Xmax: ")
        print("Ymax: ")

    # 删除表格所有行
    tabel_info_show(location_list, cls_list, conf_list, path=org_path)
    return now_img

def save_detect_image(file_path, img):#保存识别好的图片
    """保存识别后的图片"""
    if not file_path:
        print("No file to save")
        return

    if os.path.isfile(file_path):
        fileName = os.path.basename(file_path)
        name, end_name = fileName.rsplit(".", 1)
        save_name = name + '_detect_result.' + end_name
        save_img_path = os.path.join(Config.save_path, save_name)
        # 保存图片
        cv2.imwrite(save_img_path, img)
        print(f"Image saved at: {save_img_path}")
    else:
        print("Invalid file path")

def set_percent(probs):
    # 显示各表情概率值
    for i in range(len(probs)):
        print(f"Probability {i}: {round(probs[i] * 100)}%")

def tabel_info_show(locations, clses, confs, path=None):
    for location, cls, conf in zip(locations, clses, confs):
        print(f"Row: {locations.index(location) + 1}")
        print(f"Path: {path}")
        print(f"Class: {Config.CH_names[cls]}")
        print(f"Confidence: {conf}")
        print(f"Location: {location}")
        
def choose_operation():
    # 提示用户选择操作类型
    print("请选择操作类型:")
    print("0 - 打开图片")
    print("1 - 批量检测图片")
    print("2 - 打开视频")
    print("3 - 打开摄像头")
    
    # 获取用户选择
    choice = input("请输入数字(0/1/2/3): ")
    return choice

def main():
    choice = choose_operation()
    
    if choice == '0':  # 打开图片
        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './', "Image files (*.jpg *.jpeg *.png)")
        if file_path:
            open_img(file_path)

    elif choice == '1':  # 批量检测图片
        directory = QFileDialog.getExistingDirectory(None, "选取文件夹", "./")
        if directory:
            detact_batch_imgs(directory)

    elif choice == '2':  # 打开视频
        video_path = video.get_video_path()
        if video_path:
            video.video_show(video_path, model)

    elif choice == '3':  # 打开摄像头
        # 弹出对话框让用户输入摄像头索引
        index, ok = QInputDialog.getInt(None, '选择摄像头', '请输入摄像头索引（通常为0）：', 0, 0, 10, 1)
        if ok:
            print(f"[INFO] 选择的摄像头索引: {index}")
        else:
            print("[INFO] 未选择摄像头索引，使用默认索引 0")
            index = 0
        camera.camera_show(index, model)


if __name__ == "__main__":
    
    org_path = None
    fontC = ImageFont.load_default()
    main()