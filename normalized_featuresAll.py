import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def normalize_tag(tag, df):
    if df['tag'].min() != df['tag'].max():
        return (tag - df['tag'].min()) / (df['tag'].max() - df['tag'].min())
    else:
        return 1.0

def normalize_opacity(opacity):
    return opacity / 10

def normalize_color(value):
    return value / 360.0

def normalize_stroke_width(stroke_width, df):
    if df['stroke_width'].max() != df['stroke_width'].min():
        return (stroke_width - df['stroke_width'].min()) / (df['stroke_width'].max() - df['stroke_width'].min())
    else:
        return 1.0

def normalize_layer(layer):
    layer_list = eval(layer)
    if not layer_list:
        return 0.0
    max_value = max(layer_list)
    if max_value == 0:
        return 0.0
    normalized_layer_list = [x / max_value for x in layer_list]
    normalized_layer_value = sum([val * (0.1 ** idx) for idx, val in enumerate(normalized_layer_list)])
    return normalized_layer_value

def min_max_normalize(series):
    if series.min() != series.max():
        return (series - series.min()) / (series.max() - series.min())
    else:
        return pd.Series([1.0] * len(series), index=series.index)

def normalize_bbox(value):
    return np.log1p(value)

def normalize_area(value):
    return np.log1p(value)

def normalize_features(input_path, output_path):
    df = pd.read_csv(input_path)

    # 替换 -1 为 0.000001
    df.replace(-1, 0.000001, inplace=True)

    # 归一化每一列
    df['tag'] = df['tag'].apply(lambda tag: normalize_tag(tag, df))
    df['opacity'] = df['opacity'].apply(normalize_opacity)
    df['fill_h'] = df['fill_h'].map(normalize_color)
    df['stroke_h'] = df['stroke_h'].map(normalize_color)
    df[['fill_s', 'fill_l', 'stroke_s', 'stroke_l']] = df[['fill_s', 'fill_l', 'stroke_s', 'stroke_l']] / 100.0
    df['stroke_width'] = df['stroke_width'].apply(lambda sw: normalize_stroke_width(sw, df)) * 0.1
    df['layer'] = df['layer'].apply(normalize_layer)

    # 使用 Min-Max 归一化处理 bbox 相关特征
    bbox_columns = ['bbox_min_top', 'bbox_max_bottom', 'bbox_min_left', 'bbox_max_right', 'bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height']
    df[bbox_columns] = df[bbox_columns].apply(min_max_normalize)

    df['bbox_fill_area'] = df['bbox_fill_area'].map(normalize_area)
    df['bbox_stroke_area'] = df['bbox_stroke_area'].map(normalize_area)

    # 归一化后，再次应用 Min-Max 归一化，确保范围在 0 到 1 之间，并处理最小值等于最大值的情况
    df[['bbox_fill_area', 'bbox_stroke_area']] = df[['bbox_fill_area', 'bbox_stroke_area']].apply(min_max_normalize)

    # 对 bbox_stroke_area 应用权重因子，降低其影响力
    stroke_area_weight = 0.3
    df['bbox_stroke_area'] = df['bbox_stroke_area'] * stroke_area_weight

    # 保存归一化后的特征数据
    df.to_csv(output_path, index=False)

def process_all_features(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = [file_name for file_name in os.listdir(input_dir) if file_name.endswith('.csv')]

    for file_name in tqdm(files, desc="Processing files"):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        normalize_features(input_path, output_path)

# 示例使用
input_dir = './features_hsl'
output_dir = './normalized_hsl_features'
process_all_features(input_dir, output_dir)
