import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from PIL import Image
import random
import numpy as np
from sklearn.metrics import accuracy_score

# 数据集路径
data_dir = r'./dataset'
test_data_path = os.path.join(data_dir, 'test')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载测试数据集
test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)

# 创建数据加载器
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 加载模型
model = models.resnet18(weights=None)  # 使用 ResNet18
num_classes = len(test_dataset.classes)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # 添加 Dropout 层
    nn.Linear(model.fc.in_features, num_classes)
)
# 解决 torch.load 的警告问题
model.load_state_dict(torch.load('best_resnet18_model.pth', weights_only=True))  # 加载 ResNet18 的权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 随机选择20张图片
def select_random_images(test_dir, num_images=20):
    test_images = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(root, file))
    
    if len(test_images) < num_images:
        print(f"测试集文件夹中只有 {len(test_images)} 张图片，无法选择20张")
        return []
    
    return random.sample(test_images, num_images)

# 预测单张图片
def predict_image(model, image_path, transform, class_names, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)  # 将图像移动到与模型相同的设备上
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    predicted_class = class_names[predicted.item()]
    
    # 获取实际标签
    actual_label = os.path.basename(os.path.dirname(image_path))
    
    return predicted_class, actual_label, predicted_class == actual_label

# 测试函数
def test_model():
    # 1. 完整测试集评估
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # 计算整体准确率
    accuracy = accuracy_score(all_targets, all_preds)
    print(f"\n模型在测试集上的整体准确率: {accuracy:.2%}")
    
    # 2. 随机样本可视化评估
    print("\n随机样本测试结果:")
    random_images = select_random_images(test_data_path)
    if not random_images:
        return
    
    results = []
    for image_path in random_images:
        predicted, actual, is_correct = predict_image(model, image_path, transform, test_dataset.classes, device)
        results.append((os.path.basename(image_path), predicted, actual, is_correct))
    
    # 打印结果
    print("\n测试结果:")
    print(f"{'图片名称':<20} {'预测结果':<15} {'实际结果':<15} {'正确性'}")
    print("-" * 60)
    
    for name, pred, actual, correct in results:
        correct_str = "T" if correct else "F"
        print(f"{name:<20} {pred:<15} {actual:<15} {correct_str}")
    
    # 统计正确率
    correct_count = sum(1 for _, _, _, correct in results if correct)
    accuracy = correct_count / len(results)
    print(f"\n总正确率: {accuracy:.2%} ({correct_count}/{len(results)})")

if __name__ == '__main__':
    test_model()