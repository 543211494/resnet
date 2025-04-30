import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, hamming_loss
from torch.amp import GradScaler, autocast
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


matplotlib.use('TkAgg')
torch.backends.cudnn.benchmark = True
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # ResNet标准输入尺寸
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),    # 随机垂直翻转
    transforms.RandomRotation(45),      # 随机旋转45度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),   # 转为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # ImageNet标准化
])

# 数据集路径
data_dir = r'./dataset'
train_data_path = os.path.join(data_dir, 'train')
val_data_path = os.path.join(data_dir, 'val')

# 加载ImageFolder格式的数据集
train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform)

# 创建数据加载器
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 初始化 ResNet18 模型
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 修改分类层以适应自定义类别数量
# 获取类别数量
num_classes = len(train_dataset.classes)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # 添加 Dropout 层
    nn.Linear(model.fc.in_features, num_classes)
)

# 初始化权重
nn.init.xavier_uniform_(model.fc[1].weight)
model.fc[1].bias.data.fill_(0.01)

# 将模型移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 计算类别权重处理不平衡数据
class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.targets), y=train_dataset.targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# 定义损失函数、优化器和学习率调度器
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 使用混合精度训练
scaler = GradScaler()

# 训练函数
def train(model, device, train_loader, criterion, optimizer, epoch, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 清空梯度
        optimizer.zero_grad()

        # 混合精度训练上下文
        with autocast(device_type=device.__str__()):
            output = model(data)
            loss = criterion(output, target)

        # 梯度缩放和反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计信息
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 每10个batch打印一次进度
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)] '
                  f'Loss: {loss.item():.6f} | Acc: {100.*correct/total:.3f}%')

# 验证函数
def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    val_loss /= len(val_loader)
    
    metrics = {
        'loss': val_loss,
        'accuracy': accuracy_score(all_targets, all_predictions),
        'precision': precision_score(all_targets, all_predictions, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_predictions, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_predictions, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(all_targets, all_predictions),
        'hamming': hamming_loss(all_targets, all_predictions)
    }
    
    print(f"\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"{name.capitalize()}: {value:.4f}")
    
    return metrics

def main():
    try:
        epochs = 20

        print(f"当前训练使用的设备是: {device.__str__()}")

        history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'kappa': [], 'hamming': []}
        best_acc = 0

        for epoch in range(1, epochs + 1):
            torch.cuda.empty_cache()
            train(model, device, train_loader, criterion, optimizer, epoch, scaler)
            metrics = validate(model, device, val_loader, criterion)
            
            for name in history.keys():
                history[name].append(metrics[name])
            
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                torch.save(model.state_dict(), 'best_resnet18_model.pth')
            
            scheduler.step()

        # 绘图
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(['loss', 'accuracy', 'precision', 'recall', 'f1', 'kappa'], 1):
            plt.subplot(2, 3, i)
            plt.plot(history[metric])
            plt.title(metric.capitalize())
            plt.xlabel('Epoch')
            plt.ylabel('Score' if metric != 'loss' else 'Loss')
        
        plt.tight_layout()
        plt.savefig('resnet18_training_metrics.png')

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    main()