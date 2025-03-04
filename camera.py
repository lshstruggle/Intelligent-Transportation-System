# video.py
import cv2
import os
import numpy as np
import torch
import deep_sort.deep_sort.deep_sort as ds
from database import Database
import time
from PyQt5.QtWidgets import QFileDialog
from datetime import datetime
from congestion_evaluator import CongestionEvaluator  # 导入拥堵评估模块

class Config:
    CH_names = {0: "tiny-car", 1: "big-car", 2: "mid-car", 3: "small-truck", 4: "big-truck", 5: "oil-truck", 6: "special-car"}
    names = ['tiny-car', 'big-car', 'mid-car', 'small-truck', 'big-truck', 'oil-truck', 'special-car']
    save_path = "./output"



def camera_show(camera_index, model):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    save_name = 'camera_detect_result.avi'
    save_video_path = os.path.join(Config.save_path, save_name)
    out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

    cur_num = 0
    vehicle_frame_count = 0
    vehicle_class_id = 0

    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    prev_positions = {}

    db = Database(host='your_host', user='your_user', password='your_password', database='your_database')
    db.connect()

    prev_time = time.time()
    camera_id = "camera_0"  # 你可以根据需要修改摄像头ID

    # 初始化拥堵评估器
    evaluator = CongestionEvaluator(threshold=50)

    while cap.isOpened():
        cur_num += 1
        ret, frame = cap.read()
        if ret:
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            results = model(frame)[0]
            boxes = results.boxes.xyxy.tolist()
            cls_list = results.boxes.cls.tolist()

            vehicle_boxes = [box for box, cls in zip(boxes, cls_list) if cls == vehicle_class_id]
            vehicle_count = len(vehicle_boxes)
            print(f"[INFO] 当前帧中的车辆数目: {vehicle_count}")

            # 计算当前帧的平均速度
            average_speed = np.mean([np.sqrt((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) for box in vehicle_boxes]) if vehicle_boxes else 0

            # 更新拥堵评估值
            evaluator.update_congestion_value(vehicle_count, average_speed, fps)
            congestion_data = evaluator.get_congestion_data()
            prediction = evaluator.predict_congestion()

            # 在画面左上角显示车辆数目、帧率和拥堵状态
            cv2.putText(frame, f"Time: {current_datetime}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Congestion: {congestion_data['congestion_level']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            dets = []
            for box in vehicle_boxes:
                xmin, ymin, xmax, ymax = box
                dets.append([xmin, ymin, xmax, ymax, 1.0])

            if len(dets) > 0:
                dets = np.array(dets)
                tracks = tracker.update(dets[:, :4], dets[:, 4], ori_img=frame)

                for track in tracks:
                    track_id = track[4]
                    xmin, ymin, xmax, ymax = track[:4]
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2

                    speed_kmh = 0.0
                    if track_id in prev_positions:
                        prev_x, prev_y = prev_positions[track_id]
                        displacement = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                        speed_pixels_per_second = displacement * fps
                        speed_kmh = (speed_pixels_per_second * 3.6) / 20

                        print(f"车辆 {track_id} 速度: {speed_kmh:.2f} 千米/秒")

                    prev_positions[track_id] = (center_x, center_y)

                    # 插入数据到 MySQL，包括拥堵状态和预测状态
                    db.insert_vehicle_data(
                        current_datetime, camera_id, track_id, Config.names[vehicle_class_id], vehicle_count, speed_kmh,
                        congestion_data['congestion_level'], congestion_data['congestion_time'],
                        congestion_data['latest_congestion_level'], 
                        prediction['predicted_level']
                    )

            frame = results.plot()
            out.write(frame)

            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    print(f"[INFO] 总共检测到 {vehicle_frame_count} 帧包含车辆")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    db.disconnect()

