import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager as fm
import seaborn as sns
# 创建DataFrame（用你提供的数据）
# 创建 DataFrame



df = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/img/mean1.csv')
print(df)
# 1. 计算每个类别的总重要度
category_importance = df.groupby('category')['mean_importance'].sum()

# 2. 归一化类别的重要度（使类别的重要度之和为1）
category_importance_normalized = category_importance / category_importance.sum()

# 3. 归一化每个特征的重要度（每个类别内部的特征重要度可以归一化）
#df['normalized_importance'] = df.groupby('category')['mean_importance'].transform(lambda x: x / x.sum())
total_importance = df['mean_importance'].sum()
df['normalized_importance'] = df['mean_importance'] / total_importance
# 打印类别重要度和特征重要度
print("各类别的重要度：")
print(category_importance_normalized)

print("\n各特征的归一化重要度：")
print(df[['feature', 'normalized_importance']])
df.to_csv("./im.csv")
tool_color_dict = ['#4C9BE6',  # 蓝色
    '#4D4D9F',  # 浅橙色
    '#87CEEB',  # 绿色
    '#9FD1EE',  # 浅蓝色
    '#BEE8E8',  # 浅紫色
    '#E995C9',  # 青色
    '#FC9871',  # 浅粉色
    '#B0DC66',  # 灰色
    '#75C8AE',  # 浅黄色
    '#98A9D0'  # 浅棕色
]
# 对数据按 'normalized_importance' 进行排序，并选择前30个条目
df_sorted = df.sort_values(by='normalized_importance', ascending=False).head(30)
print(df_sorted[['feature', 'normalized_importance']])

# 设置字体为Times New Roman
plt.rcParams['font.size'] = 6
# 指定字体路径
font_path = '/mnt/nfs/yuht/software/ARIAL.TTF'

# 创建FontProperties对象
font_prop = FontProperties(fname=font_path)
# 将自定义字体添加到matplotlib的字体管理器中
fm.fontManager.addfont(font_path)
# 更新rcParams以使用自定义字体作为默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
# 创建图形
plt.figure(figsize=(6.6,3.3), dpi=600)

# 绘制条形图，只显示前30个特征
plt.bar(df_sorted['feature'], df_sorted['normalized_importance'], color=tool_color_dict)

# 设置X轴标签倾斜30度
plt.xticks(rotation=30, ha='right')

# 设置标签和标题
plt.xlabel('Feature', fontsize=8)
plt.ylabel('Importance', fontsize=8)
#plt.title('Top 30 Category-wise Feature Importance', fontsize=8)

# 调整布局，避免标签重叠
plt.tight_layout()

# 显示图形
plt.savefig("../img/img5/2.pdf")
plt.close()

df_cor = pd.read_csv('./test1ed.csv', low_memory=False)

new = ['phastConsElements100way', 'phyloP100way_vertebrate',
       'phyloP20way_mammalian', 'phastCons100way_vertebrate',
       'phastCons20way_mammalian', 'SiPhy_29way_logOdds',
       'phyloP30way_mammalian', 'phastCons30way_mammalian', 'AF', 'AF_raw',
       'AF_male', 'AF_female', 'AF_afr', 'AF_ami', 'AF_amr', 'AF_asj',
       'AF_eas', 'AF_fin', 'AF_nfe', 'AF_oth', 'gdi', 'gdi_phred', 'rvis1',
       'rvis2', 'lof_score',"ClinPred_score", "REVEL_score","MetaSVM_score",
       "MetaLR_score", "MutPred_score",'molecular_weight', 'equipotential_point',
       'hydrophilic', 'hydrophobic', 'amphipathic ', 'cyclic', 'essential',
       'aromatic', 'aliphatic', 'nonpolar', 'polar_uncharged', 'acidic',
       'basic', 'sulfur', 'pka_cooh', 'pka_nh3', 'blosum100', 'DS_AG', 'DS_AL',
       'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL', 'Gm12878',
       'H1hesc', 'Hepg2', 'Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf',
       ]
df_cor = df_cor[new]
categories = ['Conservation scores']*8 + [
    'Population-based features']*12 + ['Functional scores']*10+ ['Biochemical properties']*17 + ['Splicing effects']*8 + ['Epigenetic features']*9
# 计算皮尔逊相关系数
# 使用new和categories列表构建特征到类别的映射
feature_category_mapping = dict(zip(new, categories))

# 创建一个新的 DataFrame 来保存特征和类别的对应关系
df_features = pd.DataFrame({'feature': new, 'category': categories})
# 按类别对特征排序
df_features_sorted = df_features.sort_values('category')

# 重新排列 df_cor 中的特征顺序
df_cor_sorted = df_cor[df_features_sorted['feature']]

# 计算相关性矩阵
correlation_matrix = df_cor_sorted.corr()

# 获取每个类别的第一个特征的索引用于显示标签
unique_categories = df_features_sorted['category'].drop_duplicates().tolist()
y_labels = []
y_ticks = []

# 生成类别标签，只显示每个类别的第一个特征
for category in unique_categories:
    category_features = df_features_sorted[df_features_sorted['category'] == category]
    mid_idx = len(category_features) // 2  # 找到中间特征的索引
    mid_feature = category_features.iloc[mid_idx]

    y_labels.append(mid_feature['category'])
    y_ticks.append(df_features_sorted.index.get_loc(mid_feature.name))

# 绘制热图
plt.figure(figsize=(4.4, 3.3),dpi=600)

# 创建相关性热图
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

# 设置 Y 轴为类别，并使每个类别标签只显示一次
plt.yticks(ticks=y_ticks, labels=y_labels, rotation=0, fontsize=8)
plt.gca().xaxis.set_visible(False)
# 设置标题
#plt.title('Heatmap of Feature Correlations by Category (Category labels appear once)')

# 调整布局，防止标签重叠
plt.tight_layout()

# 显示图形
plt.savefig("../img/img5/1.pdf")
plt.close()