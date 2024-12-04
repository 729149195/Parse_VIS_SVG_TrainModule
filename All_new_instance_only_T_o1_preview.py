import os
import numpy as np
import torch
import json
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

def save_model(args, model, optimizer, scheduler, current_epoch):
    out = os.path.join(args.model_path, f"checkpoint_{current_epoch}.tar")
    state = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': current_epoch
    }
    torch.save(state, out)

class ContrastiveLoss(nn.Module):
    """
    带温度参数的对比损失函数。
    接受两个样本的嵌入和目标标签==1（正样本对）或标签==0（负样本对）。
    """
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, output1, output2, label):
        # 归一化嵌入向量
        output1 = F.normalize(output1, dim=1)
        output2 = F.normalize(output2, dim=1)
        # 计算相似度
        sim = self.cosine_similarity(output1, output2) / self.temperature
        # 创建标签，正样本为1，负样本为0
        labels = label
        # 计算对比损失
        # 正样本对的标签为1，负样本对的标签为0
        loss = F.binary_cross_entropy_with_logits(sim, labels)
        return loss

class Network(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Network, self).__init__()
        self.feature_dim = feature_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.feature_dim),
        ) 

    def forward(self, x):
        h = x
        z = self.instance_projector(h)
        return z

class FeaturePairDataset(Dataset):
    def __init__(self, data_dir):
        self.sample_pairs = []
        self.load_data(data_dir)

    def load_data(self, data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    all_features = [torch.tensor(feature, dtype=torch.float32) for feature in data['all_features']]
                    # 构建特征到索引的映射
                    feature_to_idx = {tuple(f.tolist()): idx for idx, f in enumerate(all_features)}
                    all_features_set = set(tuple(f.tolist()) for f in all_features)

                    groups = data['groups']

                    for group in groups:
                        if len(group) < 3:
                            continue  # 跳过数据不足的 group
                        # 最后两个元素是标识符
                        pos_identifier = group[-2]
                        neg_identifier = group[-1]
                        group_features = [torch.tensor(feature, dtype=torch.float32) for feature in group[:-2]]
                        group_feature_indices = [feature_to_idx[tuple(f.tolist())] for f in group_features]

                        # 将 group_features 转换为集合，方便后续操作
                        group_features_set = set(tuple(f.tolist()) for f in group_features)

                        # 准备负样本空间：all_features 中不在 group_features 中的特征
                        neg_sample_space = [f for f in all_features if tuple(f.tolist()) not in group_features_set]

                        # 正样本对
                        pos_pairs = []
                        pos_distances = []
                        num_pos_features = len(group_features)
                        for i in range(num_pos_features):
                            for j in range(i+1, num_pos_features):
                                f1 = group_features[i]
                                f2 = group_features[j]
                                dist = torch.dist(f1, f2).item()
                                pos_pairs.append((f1, f2))
                                pos_distances.append(dist)
                        if len(pos_pairs) == 0:
                            continue  # 无正样本对
                        # 使用 t 分布计算距离对应的概率
                        pos_probs = self.t_distribution(pos_distances)
                        # 根据标识符选择正样本对
                        pos_selected_pairs = self.select_pairs(pos_pairs, pos_probs, pos_identifier)
                        # 添加正样本对，标签为 1
                        for f1, f2 in pos_selected_pairs:
                            self.sample_pairs.append((f1, f2, 1))

                        # 负样本对
                        neg_pairs = []
                        neg_distances = []
                        for f_pos in group_features:
                            for f_neg in neg_sample_space:
                                dist = torch.dist(f_pos, f_neg).item()
                                neg_pairs.append((f_pos, f_neg))
                                neg_distances.append(dist)
                        if len(neg_pairs) == 0:
                            continue  # 无负样本对
                        # 使用 t 分布计算距离对应的概率
                        neg_probs = self.t_distribution(neg_distances)
                        # 根据标识符选择负样本对
                        neg_selected_pairs = self.select_pairs(neg_pairs, neg_probs, neg_identifier)
                        # 添加负样本对，标签为 0
                        for f1, f2 in neg_selected_pairs:
                            self.sample_pairs.append((f1, f2, 0))

    def t_distribution(self, distances):
        # distances: list of floats
        distances = np.array(distances)
        df = 1  # 自由度，可以根据需要调整
        t_probs = 1 / (1 + distances**2 / df)
        t_probs = t_probs / t_probs.sum()  # 归一化，使总和为 1
        return t_probs

    def select_pairs(self, pairs, probs, identifier):
        # pairs: list of (f1, f2)
        # probs: numpy array of probabilities
        if identifier == 3:
            percentage = 1.0
        elif identifier == 2:
            percentage = 0.75
        elif identifier == 1:
            percentage = 0.5
        else:
            percentage = 1.0  # 默认取 100%
        num_samples = int(len(probs) * percentage)
        if num_samples == 0:
            num_samples = 1  # 至少选取一个样本对
        # 获取概率最高的样本对索引
        sorted_indices = np.argsort(-probs)
        selected_indices = sorted_indices[:num_samples]
        selected_pairs = [pairs[i] for i in selected_indices]
        return selected_pairs

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, idx):
        feature1, feature2, label = self.sample_pairs[idx]
        return feature1, feature2, label

def train():
    model.train()
    total_loss = 0
    for step, (feature1, feature2, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        feature1 = feature1.to(device)
        feature2 = feature2.to(device)
        labels = labels.to(device).float()
        output1 = model(feature1)
        output2 = model(feature2)
        loss = criterion(output1, output2, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对比学习训练脚本（使用特征向量）')
    # 通用配置
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--workers', default=0, type=int, help='数据加载工作线程数')
    parser.add_argument('--data_dir', default='./DataProduce/UpdatedStepGroups', type=str, help='数据集目录')

    # 训练参数
    parser.add_argument('--batch_size', default=64, type=int, help='批大小')
    parser.add_argument('--start_epoch', default=0, type=int, help='起始epoch')
    parser.add_argument('--epochs', default=200, type=int, help='训练epoch数')

    # 模型参数
    parser.add_argument('--feature_dim', default=4, type=int, help='特征维度')
    parser.add_argument('--model_path', default='save/model', type=str, help='模型保存路径')
    parser.add_argument('--reload', action='store_true', help='从检查点重新加载模型')

    # 损失函数参数
    parser.add_argument('--learning_rate', default=0.01, type=float, help='学习率')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='权重衰减')
    parser.add_argument('--temperature', default=0.07, type=float, help='温度参数')

    # 学习率调度器参数
    parser.add_argument('--lr_scheduler', default='step', type=str, help='学习率调度器类型（例如 "step" 或 "cosine"）')
    parser.add_argument('--step_size', default=80, type=int, help='StepLR 中的 step_size')
    parser.add_argument('--gamma', default=0.1, type=float, help='StepLR 中的 gamma')
    parser.add_argument('--cosine_T_max', default=50, type=int, help='CosineAnnealingLR 中的 T_max')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备数据集
    dataset = FeaturePairDataset(args.data_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )

    if len(dataset) == 0:
        raise ValueError("数据集中没有样本对，请检查数据集。")

    input_dim = dataset.sample_pairs[0][0].size(0)

    # 初始化模型
    model = Network(input_dim, args.feature_dim)
    model = model.to(device)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    # 定义学习率调度器
    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_T_max)
    else:
        raise ValueError(f"Unsupported lr_scheduler type: {args.lr_scheduler}")

    if args.reload:
        model_fp = os.path.join(args.model_path, f"checkpoint_{args.start_epoch}.tar")
        if os.path.exists(model_fp):
            checkpoint = torch.load(model_fp)
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"从检查点 {model_fp} 重新加载模型，开始训练于 epoch {args.start_epoch}")
        else:
            print(f"检查点文件 {model_fp} 不存在，开始从头训练。")

    # 定义损失函数，加入温度参数
    criterion = ContrastiveLoss(temperature=args.temperature).to(device)

    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        print(f"开始训练 Epoch {epoch + 1}/{args.epochs}")
        loss_epoch = train()
        scheduler.step()  # 更新学习率
        if (epoch + 1) % 5 == 0:
            save_model(args, model, optimizer, scheduler, epoch + 1)
            print(f"已保存模型至 epoch {epoch + 1}")
        print(f"Epoch [{epoch + 1}/{args.epochs}]\t Loss: {loss_epoch:.4f}\t 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

    # 保存最终模型
    save_model(args, model, optimizer, scheduler, args.epochs)
    print("训练完成，最终模型已保存。")
