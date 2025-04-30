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

# 计算类别权重（与原始代码相同）
from sklearn.utils.class_weight import compute_class_weight


matplotlib.use('TkAgg')
torch.backends.cudnn.benchmark = True
# 数据预处理（与原始代码相同）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 数据集路径（与原始代码相同）
data_dir = r'./dataset'
train_data_path = os.path.join(data_dir, 'train')
val_data_path = os.path.join(data_dir, 'val')

# 加载数据集（与原始代码相同）
train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform)

# 创建数据加载器（与原始代码相同）
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 初始化 ResNet18 模型（关键修改点 1）
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 修改分类层（关键修改点 2）
num_classes = len(train_dataset.classes)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # 添加 Dropout 层
    nn.Linear(model.fc.in_features, num_classes)
)
nn.init.xavier_uniform_(model.fc[1].weight)  # 初始化权重
model.fc[1].bias.data.fill_(0.01)

# 将模型移动到 GPU（与原始代码相同）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.targets), y=train_dataset.targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# 定义损失函数、优化器和学习率调度器（与原始代码相同）
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 使用混合精度训练（与原始代码相同）
scaler = GradScaler()

# 训练函数（与原始代码相同）
def train(model, device, train_loader, criterion, optimizer, epoch, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        with autocast(device_type=device.__str__()):
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)] '
                  f'Loss: {loss.item():.6f} | Acc: {100.*correct/total:.3f}%')

# 验证函数（与原始代码相同）
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

# 主训练流程（修改模型保存名称）
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
                torch.save(model.state_dict(), 'best_resnet18_model.pth')  # 修改保存名称
            
            scheduler.step()

        # 绘图（与原始代码相同）
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(['loss', 'accuracy', 'precision', 'recall', 'f1', 'kappa'], 1):
            plt.subplot(2, 3, i)
            plt.plot(history[metric])
            plt.title(metric.capitalize())
            plt.xlabel('Epoch')
            plt.ylabel('Score' if metric != 'loss' else 'Loss')
        
        plt.tight_layout()
        plt.savefig('resnet18_training_metrics.png')  # 修改图片名称

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    main()