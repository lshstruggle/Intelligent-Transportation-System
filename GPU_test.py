import torch
print("CUDA Available:", torch.cuda.is_available())  # 是否可用
print("CUDA Version:", torch.version.cuda)  # CUDA 版本
print("CUDA Device Count:", torch.cuda.device_count())  # 可用的 GPU 数量
