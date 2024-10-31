import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['axes.unicode_minus'] = False

# 定义输入特征名称
feature_names = [
    "tag", "opacity", "fill_h_cos", "fill_h_sin", "fill_s_n", "fill_l_n",
    "stroke_h_cos", "stroke_h_sin", "stroke_s_n", "stroke_l_n", "stroke_width",
    "layer_sal", "bbox_left_n", "bbox_right_n", "bbox_top_n", "bbox_bottom_n",
    "bbox_center_x_n", "bbox_center_y_n", "bbox_width_n", "bbox_height_n",
    "bbox_fill_area", "bbox_stroke_area"
]
num_features = len(feature_names)  # 输入特征数

# 加载保存的权重
with open('model_weights.pkl', 'rb') as f:
    loaded_weights = pickle.load(f)

# 初始化存储每层权重的列表和层名称
aggregated_weights = []
layer_names = []

# 遍历每一层的权重，检查是否与特征数匹配
for layer_name, weights in loaded_weights.items():
    if 'weight' in layer_name:
        if weights.shape[1] == num_features:  # 如果列数与特征数匹配
            avg_weights = np.mean(np.abs(weights), axis=0)  # 计算每列的平均绝对值
            aggregated_weights.append(avg_weights)
            layer_names.append(layer_name)
        else:
            print(f"警告：跳过{layer_name}，因为其权重大小不匹配输入特征数。")

# 将所有层的权重数据组合成一个大矩阵
weights_matrix = np.vstack(aggregated_weights)  # 将各层权重堆叠成矩阵

# 绘制热力图
plt.figure(figsize=(15, 10))
sns.heatmap(weights_matrix.T, annot=True, fmt=".2f", yticklabels=feature_names, xticklabels=layer_names, cmap="Blues")
plt.ylabel("Input Features")
plt.xlabel("Model Layers")
plt.title("Weight Distribution Across Features for Each Model Layer")
plt.show()
