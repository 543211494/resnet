import os
import shutil
import splitfolders

# 蘑菇类别映射字典
mushroom_mapping = {
    'class00': '羊肚菌',
    'class01': '牛肝菌',
    'class02': '鸡油菌',
    'class03': '鸡枞菌',
    'class04': '青头菌',
    'class05': '奶浆菌',
    'class06': '干巴菌',
    'class07': '虎掌菌',
    'class08': '白葱牛肝菌',
    'class09': '老人头菌',
    'class10': '猪肚菌',
    'class11': '谷熟菌',
    'class12': '白参菌',
    'class13': '黑木耳',
    'class14': '银耳',
    'class15': '金耳',
    'class16': '猴头菇',
    'class17': '香菇',
    'class18': '平菇',
    'class19': '金针菇',
    'class20': '口蘑',
    'class21': '鹿茸菇',
    'class22': '榆黄蘑',
    'class23': '榛蘑',
    'class24': '草菇',
    'class25': '鸡腿菇',
    'class26': '茶树菇',
    'class27': '蟹味菇',
    'class28': '白玉菇',
    'class29': '红菇',
    'class30': '杏鲍菇',
    'class31': '松茸',
    'class32': '姬松茸',
    'class33': '松露',
    'class34': '竹荪',
    'class35': '虫草花'
}

# 原始数据路径和输出路径
data_dir = r'./data/archive/data'
output_dir = r'./dataset'

# 1. 先重命名文件夹
for old_name, new_name in mushroom_mapping.items():
    old_path = os.path.join(data_dir, old_name)
    new_path = os.path.join(data_dir, new_name)
    
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f'已重命名: {old_name} -> {new_name}')
    else:
        print(f'警告: 文件夹 {old_name} 不存在')

# 2. 划分数据集
print("\n开始划分数据集...")
splitfolders.ratio(
    data_dir,
    output=output_dir,
    seed=1337,
    ratio=(0.8, 0.1, 0.1),  # 训练集80%，验证集10%，测试集10%
    group_prefix=None,
    move=False
)

print('\n所有操作完成！数据集已划分到:', output_dir)