import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, jaccard_score, matthews_corrcoef, auc, average_precision_score, precision_recall_fscore_support, precision_recall_curve, roc_curve
# Load the data test2ed_del

print("加载数据")
print("加载数据")
# IRDTest
#df = pd.read_csv('./delete/test3ed.csv', low_memory=False)  # 示例：从CSV文件中读取数据
# SwissProt-Test
#df = pd.read_csv('./delete/test2ed.csv', low_memory=False)  # 示例：从CSV文件中读取数据
# VarRareTest
#df = pd.read_csv('./test1ed_rare.csv', low_memory=False)
#CVTesting
df = pd.read_csv('./test1ed.csv', low_memory=False)
#Traning
df1 = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/train/train.csv', low_memory=False)
#val
df2 = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/train/val.csv', low_memory=False)
#df = pd.concat([df1, df2], axis=0, ignore_index=True)
df['CLASS'] = df['CLASS'].replace(-1, 0)
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
# 使用value_counts()获取类别及其计数
value_counts = df['CLASS'].value_counts()

# 将数字映射为描述性标签
category_map = {1: 'Pathogenic', 0: 'Benign'}
categories = [category_map[idx] if idx in category_map else idx for idx in value_counts.index]
counts = value_counts.tolist()
tool_color_dict = {
    'GNN-MAP': '#4C9BE6',  # 蓝色
    'REVEL': '#4D4D9F',  # 浅橙色
    'PrimateAI': '#87CEEB',  # 绿色
    'MutPred': '#9FD1EE',  # 浅蓝色
    'MetaSVM': '#BEE8E8',  # 浅紫色
    'VEST4': '#E995C9',  # 青色
    'MetaLR': '#FC9871',  # 浅粉色
    'M-CAP': '#B0DC66',  # 灰色
    'ClinPred': '#75C8AE',  # 浅黄色
    'VARITY_R': '#98A9D0'  # 浅棕色
}
# 调整后的颜色列表
colors11 = ['#FBCB1F', '#F2F3C7']
colors = ['#87CEEB', '#BEE8E8']
colors12 = ['#1C6AB1', '#0D95CE', '#5DBDE8', '#89BCE4']
colors1 = ['#B0DC66', '#FC9871', '#98A9D0','#75C8AE','#D097BA','#98A9D0']

# 函数来格式化饼图中的文本显示为计数
def absolute_value(val, allvals):
    absolute = int(round(val / 100. * sum(allvals)))
    return f"{absolute}"

# 生成饼图
plt.figure(figsize=(2.2,1.1),dpi=600)
plt.pie(counts, labels=categories, autopct=lambda p: absolute_value(p, counts), startangle=140, colors=colors)
#plt.text(-0.1, 1.05, 'A', fontproperties=font_prop, fontsize=20, va='top', ha='right')
plt.savefig("../img/img1/wen/pie_chart_1.png")
plt.close()
#gai
columns_to_plot = [
     'func_nonsynonymous SNV','func_stopgain','func_frameshift','func_startloss','func_nonframeshift',
'func_stoploss'
]

# 创建不同类型标签
labels = ['Missence', 'Stopgain', 'Frameshift', 'Startloss', 'Nonframeshift','Stoploss']

# 统计每列中 1 的总数
counts3 = [df[col].sum() for col in columns_to_plot]
counts4 = [df[df['CLASS'] == 0][col].sum() for col in columns_to_plot]

print(counts4)
# 设置颜色
# 生成条形图
plt.figure(figsize=(2.2, 1.1), dpi=600)
plt.barh(labels, counts3, color=colors1)

# 设置标签和标题
plt.xlabel('Variation Count', fontsize=8)
plt.ylabel('Variation Type', fontsize=8)

# 显示百分比
for i, count in enumerate(counts3):
    plt.text(count + 300, i, f'{count / sum(counts3) * 100:.1f}%', ha='left', va='center')

# 隐藏边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


# 美化图表布局
plt.tight_layout()

# 保存图表
plt.savefig("../result_img/fig3/data_stat/bar_chart_1.png")
plt.close()

