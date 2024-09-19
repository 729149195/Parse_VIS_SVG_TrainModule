import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import color
import numpy as np

# 读取特征数据
features_path = './features_Lab/371.csv'
df = pd.read_csv(features_path)

# 创建绘图
fig, ax = plt.subplots(figsize=(10, 15))

# 计算整体图形的高度，以便正确地从左上角开始绘制
overall_height = df['bbox_max_bottom'].max() + 50

# 绘制每个矩形
for _, row in df.iterrows():
    # 填充颜色转换（Lab）
    fill_l, fill_a, fill_b = row['fill_l'], row['fill_a'], row['fill_b']
    if fill_l >= 0 and fill_a >= 0 and fill_b >= 0:
        fill_color = np.clip(color.lab2rgb([[[fill_l, fill_a, fill_b]]]), 0, 1).flatten()
    else:
        fill_color = (1, 1, 1)  # 默认为白色

    # 边框颜色转换（Lab）
    stroke_l, stroke_a, stroke_b = row['stroke_l'], row['stroke_a'], row['stroke_b']
    if stroke_l >= 0 and stroke_a >= 0 and stroke_b >= 0:
        stroke_color = np.clip(color.lab2rgb([[[stroke_l, stroke_a, stroke_b]]]), 0, 1).flatten()
    else:
        stroke_color = (0, 0, 0)  # 默认为黑色

    # 绘制矩形
    rect = patches.Rectangle(
        (row['bbox_min_left'], overall_height - row['bbox_max_bottom']),
        row['bbox_width'],
        row['bbox_height'],
        linewidth=row['stroke_width'],
        edgecolor=stroke_color,
        facecolor=fill_color,
        alpha=row['opacity']
    )
    ax.add_patch(rect)

# 设置绘图范围
ax.set_xlim(0, df['bbox_max_right'].max() + 50)
ax.set_ylim(0, overall_height)

plt.show()
