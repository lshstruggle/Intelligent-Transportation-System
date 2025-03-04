import os
import torch
from ultralytics import YOLO

# 数据集路径
data_path = '/root/car-detection/data.yaml'

if __name__ == '__main__':
    # 设置环境变量来避免显存碎片化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # 加载模型
    model = YOLO('yolov10m.yaml')

    # 清理显存
    torch.cuda.empty_cache()

    # 训练
    results = model.train(
        data=data_path,
        epochs=1000,
        batch=32,              # 调整为32
        imgsz=640,             # 调整为640
        device="0",            # 使用GPU 0
        workers=8,
        project='runs/detect',
        name='exp',
        half=False,            # 禁用混合精度训练
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        save_period=10,
        augment=True,
        dropout=0.1,
        patience=30,
        verbose=True
    )
    
    # 训练完后清理显存
    torch.cuda.empty_cache()