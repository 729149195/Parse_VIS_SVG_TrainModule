import os
import colorsys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 定义特征文件夹路径
features_dir = './features_v2'

# 从 1 开始，每隔 10 个数读取文件
start = 1
step = 10

# 循环处理每隔10个数的文件 1_features.csv, 11_features.csv, 21_features.csv 等
for i in range(start, 101, step):  # 假设最多到 100_features.csv，可以根据具体文件数量调整
    features_path = os.path.join(features_dir, f'{i}_features.csv')
    
    # 检查文件是否存在，避免缺少文件时报错
    if not os.path.exists(features_path):
        print(f"File {features_path} does not exist. Skipping...")
        continue

    # 读取特征数据
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

    # 显示或保存绘制的图像
    plt.title(f"Visualization of {i}_features.csv")
    plt.show()

    # 如果想保存图像而不是显示，替换 plt.show() 为以下代码
    # output_image_path = f"./output_images/{i}_features_visualization.png"
    # plt.savefig(output_image_path)
    # print(f"Saved image to {output_image_path}")

    plt.close(fig)  # 关闭当前图形以避免内存泄漏
