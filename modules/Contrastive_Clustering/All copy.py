import torch
import math
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
import random

epochs = 200
temperature = 0.05  
batch_size = 128 
learning_rate = 0.001  
dataset_path = "./n_v5_features"
model_save_path = "save/model_checkpoint_class20_llrr_v12.tar"  
train_losses = []
eval_losses = []


if not os.path.exists(os.path.dirname(model_save_path)):
    os.makedirs(os.path.dirname(model_save_path))

class InstanceLoss(nn.Module):
    def __init__(self, temperature, device):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)
        N = z_i.size(0)
        sim = torch.mm(z, z.T) / self.temperature
        sim.fill_diagonal_(-float('inf'))
        labels = torch.arange(N).to(self.device)
        labels = torch.cat([labels, labels], dim=0)
        positives = torch.cat([torch.diag(sim[:N, N:]), torch.diag(sim[N:, :N])], dim=0)
        mask = ~torch.eye(2 * N, dtype=bool).to(self.device)
        negatives = sim[mask].view(2 * N, -1)
        logits = torch.cat((positives.unsqueeze(1), negatives), dim=1)
        labels = torch.zeros(2 * N, dtype=torch.long).to(self.device)
        loss = self.criterion(torch.log_softmax(logits, dim=1), labels)
        return loss.mean()

class DynamicClusterLoss(nn.Module):
    def __init__(self, temperature, device):
        super(DynamicClusterLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum() + 1e-18  
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i + 1e-8)).sum()  
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum() + 1e-18  
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j + 1e-8)).sum()  
        ne_loss = ne_i + ne_j
        c_i = c_i.t()
        c_j = c_j.t()
        class_num = c_i.size(0)
        N = 2 * class_num
        c = torch.cat((c_i, c_j), dim=0)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, class_num)
        sim_j_i = torch.diag(sim, -class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim.masked_select(~torch.eye(N, dtype=torch.bool).to(self.device)).reshape(N, -1)
        labels = torch.zeros(N).to(self.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(torch.log_softmax(logits, dim=1), labels)
        loss /= N
        return loss + ne_loss


class ModifiedNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim, class_num):
        super(ModifiedNetwork, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.cluster_num),
            nn.Softmax(dim=1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = normalize(self.instance_projector(x), dim=1)
        c = self.cluster_projector(x)
        return z, c

    def forward_cluster(self, x):
        c = self.cluster_projector(x)
        c = torch.argmax(c, dim=1)
        return c


class FeaturePairDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.files = os.listdir(directory)
        self.features = []
        self.expected_feature_length = 22  # 期望的特征长度
        self.load_features()
        
        import random

    # 添加基于随机平移的正样本
    def add_random_shift_samples(self, features):
        shifted_samples = []
        for _ in range(20):  # 加20个增强样本
            # 随机平移比例的10%
            shift_x = random.uniform(-0.1, 0.1) * 2 * features[18]  # bbox_max_right, bbox_min_left
            # shift_y = random.uniform(-0.1, 0.1) * features[12]  # bbox_max_bottom, bbox_min_top

            # 生成上下左右平移后的样本
            shifted_sample = features.copy()
            # shifted_sample[12] += shift_y
            # shifted_sample[13] += shift_y
            shifted_sample[14] += shift_x
            shifted_sample[15] += shift_x
            shifted_sample[17] += shift_x
            shifted_samples.append(shifted_sample)

        return shifted_samples


    def load_features(self):
        for file in self.files:
            file_path = os.path.join(self.directory, file)
            df = pd.read_csv(file_path)
            for index, row in df.iterrows():
                features = [float(row.iloc[i]) for i in range(1, 23) if not pd.isnull(row.iloc[i])]
                if len(features) != self.expected_feature_length:
                    print(f"Unexpected feature length in file {file} at row {index}: {len(features)}")
                    continue
                self.features.append(features)
                
                # 添加基于接近性的正样本
                shifted_samples = self.add_random_shift_samples(features)
                self.features.extend(shifted_samples)
                
                #生成整段平移的正样本
                # flipped_next_left = features.copy()
                # flipped_next_left[12], flipped_next_left[13], flipped_next_left[16] = features[12] + 2 * features[18], features[13] + 2 * features[18], features[16] + 2 * features[18]
                # self.features.append(flipped_next_left)
                
                # flipped_before_right = features.copy()
                # flipped_before_right[12], flipped_before_right[13], flipped_before_right[16] = features[12] - 2 * features[18], features[13] - 2 * features[18], features[16] - 2 * features[18]
                # self.features.append(flipped_before_right)
                
                
                # flipped_next_left = features.copy()
                # flipped_next_left[14], flipped_next_left[15], flipped_next_left[17] = features[14] + 2 * features[18], features[15] + 2 * features[18], features[17] + 2 * features[18]
                # self.features.append(flipped_next_left)
                
                # flipped_before_right = features.copy()
                # flipped_before_right[14], flipped_next_left[15], flipped_next_left[17] = features[14] - 2 * features[18], features[15] - 2 * features[18], features[17] - 2 * features[18]
                # self.features.append(flipped_before_right)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32)



def simplified_train_loop(dataset, model, instance_loss, cluster_loss, optimizer, epochs=epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            features = features.to(device)
            optimizer.zero_grad()
            z, c = model(features)

            # 使用正样本对进行损失计算
            loss_i = instance_loss(z, z)  # 使用增强样本生成的z进行计算
            loss_c = cluster_loss(c, c)    # 使用增强样本生成的c进行计算
            
            if torch.isnan(loss_i) or torch.isnan(loss_c):
                print("Nan loss encountered. Skipping the current batch.")
                continue

            loss = loss_i + loss_c
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(DataLoader(dataset))}")

        eval_loss = your_eval_function(model, eval_dataset)
        eval_losses.append(eval_loss)



def save_model(model, optimizer, epoch, save_path):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_path)


def load_model(model, optimizer, load_path):
    checkpoint = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def plot_losses(train_losses, eval_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    if eval_losses:
        plt.plot(eval_losses, label='Eval Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Eval Loss')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    input_dim = 22
    feature_dim = 22
    class_num = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedNetwork(input_dim, feature_dim, class_num).to(device)
    instance_loss = InstanceLoss(temperature=temperature, device=device)
    cluster_loss = DynamicClusterLoss(temperature=temperature, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = FeaturePairDataset(dataset_path)
    start_epoch = 0
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {start_epoch}.")
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for features in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            features = features.to(device)
            optimizer.zero_grad()
            z, c = model(features)
            loss_i = instance_loss(z, z) 
            loss_c = cluster_loss(c, c) 
            loss = loss_i + loss_c
            if torch.isnan(loss):
                print("Nan loss encountered. Skipping the current batch.")
                print(f"z: {z}")
                print(f"c: {c}")
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(DataLoader(dataset))}")
        save_model(model, optimizer, epoch, model_save_path)
    
    plot_losses(train_losses, eval_losses)
