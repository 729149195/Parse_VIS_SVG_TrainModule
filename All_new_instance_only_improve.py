import os
import numpy as np
import torch
import json
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from sklearn.cluster import AgglomerativeClustering

def save_model(args, model, optimizer, scheduler, current_epoch):
    out = os.path.join(args.model_path, f"checkpoint_{current_epoch}.tar")
    state = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': current_epoch
    }
    torch.save(state, out)

class ImprovedInstanceLoss(nn.Module):
    def __init__(self, device, temperature=1, similarity_threshold=0.5):
        super(ImprovedInstanceLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold

    def compute_similarity(self, features): # 根据多个维度计算相似度（计算各个特征之间的欧几里得距离）
        # 标签相似性
        tag_sim = torch.cdist(features[:, 0:1], features[:, 0:1], p=2)
        tag_sim = 1 - (tag_sim / tag_sim.max())

        # 透明度相似性
        opacity_sim = torch.cdist(features[:, 1:2], features[:, 1:2], p=2)
        opacity_sim = 1 - (opacity_sim / opacity_sim.max())

        # 颜色相似性 - 填充色相
        fill_color_h_sim = torch.cdist(features[:, 2:4], features[:, 2:4], p=2)
        fill_color_h_sim = 1 - (fill_color_h_sim / fill_color_h_sim.max())

        # 颜色相似性 - 填充饱和度和亮度
        fill_color_sl_sim = torch.cdist(features[:, 4:6], features[:, 4:6], p=2)
        fill_color_sl_sim = 1 - (fill_color_sl_sim / fill_color_sl_sim.max())

        # 颜色相似性 - 描边色相
        stroke_color_h_sim = torch.cdist(features[:, 6:8], features[:, 6:8], p=2)
        stroke_color_h_sim = 1 - (stroke_color_h_sim / stroke_color_h_sim.max())

        # 颜色相似性 - 描边饱和度和亮度
        stroke_color_sl_sim = torch.cdist(features[:, 8:10], features[:, 8:10], p=2)
        stroke_color_sl_sim = 1 - (stroke_color_sl_sim / stroke_color_sl_sim.max())

        # 描边宽度相似性
        stroke_width_sim = torch.cdist(features[:, 10:11], features[:, 10:11], p=2)
        stroke_width_sim = 1 - (stroke_width_sim / stroke_width_sim.max())

        # 图层显著性相似性
        layer_sal_sim = torch.cdist(features[:, 11:12], features[:, 11:12], p=2)
        layer_sal_sim = 1 - (layer_sal_sim / layer_sal_sim.max())

        # 边界框坐标相似性
        bbox_coords_sim = torch.cdist(features[:, 12:16], features[:, 12:16], p=2)
        bbox_coords_sim = 1 - (bbox_coords_sim / bbox_coords_sim.max())

        # 边界框中心位置相似性
        bbox_center_sim = torch.cdist(features[:, 16:18], features[:, 16:18], p=2)
        bbox_center_sim = 1 - (bbox_center_sim / bbox_center_sim.max())

        # 边界框尺寸相似性
        bbox_size_sim = torch.cdist(features[:, 18:20], features[:, 18:20], p=2)
        bbox_size_sim = 1 - (bbox_size_sim / bbox_size_sim.max())

        # 边界框填充面积相似性
        bbox_fill_area_sim = torch.cdist(features[:, 20:21], features[:, 20:21], p=2)
        bbox_fill_area_sim = 1 - (bbox_fill_area_sim / bbox_fill_area_sim.max())

        # 综合相似性计算（可以根据需要调整权重）
        combined_sim = (
            tag_sim + 
            opacity_sim + 
            fill_color_h_sim + 
            fill_color_sl_sim + 
            stroke_color_h_sim + 
            stroke_color_sl_sim + 
            stroke_width_sim + 
            layer_sal_sim + 
            bbox_coords_sim + 
            bbox_center_sim + 
            bbox_size_sim + 
            bbox_fill_area_sim
        ) / 12
        return combined_sim

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        similarity_matrix = self.compute_similarity(features)
        
        # 使用层次聚类对相似元素进行分组
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=self.similarity_threshold)
        cluster_labels = clustering.fit_predict(similarity_matrix.cpu().detach().numpy())
        
        batch_size = labels.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool).to(self.device)
        
        # 根据聚类结果创建正掩码
        positive_mask = torch.tensor([[cluster_labels[i] == cluster_labels[j] for j in range(batch_size)] for i in range(batch_size)], dtype=torch.bool).to(self.device)
        
        # 负掩码是指所有非正掩码对
        negative_mask = ~positive_mask & ~mask
        
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        sim_matrix_exp = torch.exp(sim_matrix)
        sim_matrix_exp = sim_matrix_exp * (~mask)

        # 正样本对相似度
        pos_sim = (sim_matrix_exp * positive_mask.float()).sum(dim=1)
        # 负样本对相似度
        neg_sim = (sim_matrix_exp * negative_mask.float()).sum(dim=1)

        pos_div = pos_sim / (pos_sim + neg_sim + 1e-8)
        pos_div = torch.clamp(pos_div, min=1e-8)
        loss = -torch.log(pos_div)
        loss = loss.mean()
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
        z = F.normalize(self.instance_projector(h), dim=1)
        return z

class FeatureVectorDataset(Dataset):
    def __init__(self, data_dir):
        self.all_features = []
        self.labels = []
        self.load_data(data_dir)

    def load_data(self, data_dir):
        unique_label = 0  # 初始化唯一标签计数器
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    all_features = data['all_features']
                    groups = data['groups']
                    # 转换特征为张量
                    features = [torch.tensor(feature, dtype=torch.float32) for feature in all_features]
                    feature_assigned = [False] * len(features)
                    # 构建特征到索引的映射
                    feature_to_idx = {idx: features[idx] for idx in range(len(features))}
                    # 处理组
                    for group in groups:
                        # 获取组内特征对应的索引
                        group_indices = []
                        for feature in group:
                            for idx, f in enumerate(features):
                                if torch.allclose(f, torch.tensor(feature, dtype=torch.float32)):
                                    group_indices.append(idx)
                                    break
                        # 根据第14维排序
                        group_indices_sorted = sorted(group_indices, key=lambda idx: features[idx][12].item())
                        group_indices_sorted = sorted(group_indices_sorted, key=lambda idx: features[idx][13].item())
                        # 将排序后的特征添加到数据集中
                        for idx in group_indices_sorted:
                            if not feature_assigned[idx]:
                                self.all_features.append(features[idx])
                                self.labels.append(unique_label)
                                feature_assigned[idx] = True
                        unique_label += 1
                    # 处理未分配的特征
                    for idx, assigned in enumerate(feature_assigned):
                        if not assigned:
                            self.all_features.append(features[idx])
                            self.labels.append(unique_label)
                            unique_label += 1
        # 将标签转换为张量
        self.labels = torch.tensor(self.labels, dtype=torch.long)


    def __len__(self):
        return len(self.all_features)

    def __getitem__(self, idx):
        feature = self.all_features[idx]
        label = self.labels[idx]
        return feature, label

def train():
    model.train()
    total_loss = 0
    for step, (features, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        features = features.to(device)
        labels = labels.to(device)
        z = model(features)
        loss_instance = criterion_instance(z, labels)
        loss = loss_instance
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # if step % 10 == 0:
        #     print(f"Step [{step}/{len(data_loader)}]\t Loss_instance: {loss_instance.item():.4f}")
    return total_loss / len(data_loader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对比学习训练脚本（使用特征向量）')
    # 通用配置
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--workers', default=0, type=int, help='数据加载工作线程数')
    parser.add_argument('--data_dir', default='./UpdatedStepGroups', type=str, help='数据集目录')

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
    dataset = FeatureVectorDataset(args.data_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True
    )

    input_dim = dataset.all_features[0].size(0)

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

    criterion_instance = ImprovedInstanceLoss(device).to(device)

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
