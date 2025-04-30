import shutil

import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("huizecai/mushroom")
print("dataset path:",path)

# 要存放数据集的目标路径
target_dir = './data'

os.makedirs(target_dir, exist_ok=True)

# 移动数据集
for file_name in os.listdir(path):
    shutil.move(os.path.join(path, file_name), os.path.join(target_dir, file_name))

print("Files have been moved to:", target_dir)