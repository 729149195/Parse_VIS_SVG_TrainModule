import colorsys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 读取特征数据
features_path = './features_hsl/371.csv'
df = pd.read_csv(features_path)

# 创建绘图
fig, ax = plt.subplots(figsize=(10, 15))

# 计算整体图形的高度，以便正确地从左上角开始绘制
overall_height = df['bbox_max_bottom'].max() + 50

# 绘制每个矩形
for _, row in df.iterrows():
    # 填充颜色转换
    fill_h, fill_s, fill_l = row['fill_h'], row['fill_s'], row['fill_l']
    if fill_h >= 0 and fill_s >= 0 and fill_l >= 0:
        fill_color = colorsys.hls_to_rgb(fill_h / 360, fill_l / 100, fill_s / 100)
    else:
        fill_color = (1, 1, 1)  # 默认为白色

    # 边框颜色转换
    stroke_h, stroke_s, stroke_l = row['stroke_h'], row['stroke_s'], row['stroke_l']
    if stroke_h >= 0 and stroke_s >= 0 and stroke_l >= 0:
        stroke_color = colorsys.hls_to_rgb(stroke_h / 360, stroke_l / 100, stroke_s / 100)
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

# ax.invert_yaxis()

plt.show()
