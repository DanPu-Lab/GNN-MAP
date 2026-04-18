
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, jaccard_score, matthews_corrcoef, auc, average_precision_score, precision_recall_fscore_support, precision_recall_curve, roc_curve
# Load the data test2ed_del
print("加载数据")
df = pd.read_csv('./delete/test3ed.csv', low_memory=False)  # 示例：从CSV文件中读取数据
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
plt.figure(figsize=(2.2,1.1),dpi=100)
plt.pie(counts, labels=categories, autopct=lambda p: absolute_value(p, counts), startangle=140, colors=colors)
#plt.text(-0.1, 1.05, 'A', fontproperties=font_prop, fontsize=20, va='top', ha='right')
#plt.savefig("../img/img1/pie_chart2.pdf")
plt.show()
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
#plt.savefig("../img/img1/bar_chart1.pdf")
plt.close()

# 加载数据
df = pd.read_csv('./test1ed.csv')
num = df['AF']
df['CLASS'] = df['CLASS'].replace(-1, 0)
num_sum = np.sum(num == 0)
df['GNN-MAP'] = df['GNN-MAP_score']
# 列出预测列
predprod_columns = ['MetaSVM','MetaLR','M-CAP','VARITY_R',
    'MutPred','ClinPred','VEST4','REVEL','PrimateAI',
    'GNN-MAP']

# 创建字典保存AUC分数
auc_scores = {}
auprc_scores = {}
metrics_dict = {}
prediction_thresholds = {
    'MetaSVM': 0.5,
    'MetaLR': 0.5,
    'M-CAP': 0.025,
    'VARITY_R': 0.5,
    'MutPred': 0.79,
    'ClinPred': 0.5,
    'VEST4': 0.5,
    'REVEL': 0.5,
    'PrimateAI': 0.8,
    'GNN-MAP': 0.5,
}

# 计算每个工具的错误率并存储
for column in predprod_columns:
    data = df.dropna(subset=[column, 'Chr'])
    # True labels
    y_values = data['CLASS'].values
    pre_value = data[column].values  # Extract the prediction values for the current column
    fpr, tpr, thresholds = roc_curve(y_values, pre_value)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = prediction_thresholds[column]
    pre_value = (pre_value > optimal_threshold).astype(int)

    try:
        metrics = {}
        # Calculate each metric and store it in the dictionary
        metrics['Accuracy'] = accuracy_score(y_values, pre_value)
        metrics['Precision'], metrics['Recall'], metrics['F1 Score'], _ = precision_recall_fscore_support(y_values, pre_value, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_values, pre_value).ravel()
        metrics['Specificity'] = tn / (tn + fp)
        metrics['G-Mean'] = np.sqrt(metrics['Recall'] * metrics['Specificity'])
        metrics['MCC'] = matthews_corrcoef(y_values, pre_value)
        num_all = df.shape[0]
        num_notnull = sum(df[column].isnull())
        metrics['Missing Rate'] = num_notnull / num_all
        # Add the dictionary of metrics for the current column to the main dictionary
        metrics_dict[column] = metrics
    except ValueError as e:
        # Handle cases where AUC cannot be computed
        print(f'Could not compute metrics for {column}: {e}')

print(f"{'Tool':<15}", end="")
for metric_name in metrics_dict[next(iter(metrics_dict))]:
    print(f"{metric_name:<15}", end="")
print()
for column, metrics in metrics_dict.items():
    print(f"{column:<15}", end="")
    for metric_value in metrics.values():
        print(f"{metric_value:<15.4f}", end="")
    print()

# 按错误率排序
sorted_metrics = sorted(metrics_dict.items(), key=lambda item: item[1]['Missing Rate'])
for column in predprod_columns:
    data = df.dropna(subset=column)
    data = data.dropna(subset='Chr')
    # True labels
    y_values = data['CLASS'].values
    pred_prob = data[column].values  # Extract the prediction values for the current column
    try:
        auc_score1 = roc_auc_score(y_values, pred_prob)
        precision_curve1, recall_curve1, _ = precision_recall_curve(y_values, pred_prob)
        auprc1 = auc(recall_curve1, precision_curve1)
        auprc_scores[column] = auprc1
        auc_scores[column] = auc_score1
    except ValueError as e:
        # Handle cases where AUC cannot be computed
        print(f'Could not compute AUC for {column}: {e}')

# 生成ROC曲线图
plt.figure(figsize=(2.2,2.2),dpi=600)
sorted_columns = sorted(auc_scores, key=auc_scores.get, reverse=True)
for column in sorted_columns:
    data = df.dropna(subset=column)
    data = data.dropna(subset='Chr')
    # True labels
    y_values = data['CLASS'].values
    pre_value = data[column].values
    fpr, tpr, thresholds = roc_curve(y_values, pre_value)
    auc_score = auc_scores[column]
    cols = column.replace('_score', '')
    cols = cols.replace('_rankscore', '')
    color = tool_color_dict.get(cols, 'skyblue')  # 默认为'skyblue'，如果没有找到对应的颜色
    plt.plot(fpr, tpr, label=f'{cols} ({auc_score:.3f})', color=color)
plt.plot([0, 1], [0, 1], 'k--', label='Chance')  # Add a diagonal dashed line
#plt.text(-0.1, 1.05, 'C', fontproperties=font_prop, fontsize=20, va='top', ha='right')
plt.xlabel('False positive rate', fontproperties=font_prop, fontsize=8)
plt.ylabel('True positive rate', fontproperties=font_prop, fontsize=8)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("../result_img/img3/roc_curve1.png")
plt.close()

# 生成Precision-Recall曲线图
plt.figure(figsize=(2.2,2.2),dpi=600)
sorted_columns1 = sorted(auprc_scores, key=auprc_scores.get, reverse=True)
for column in sorted_columns1:
    data = df.dropna(subset=column)
    data = data.dropna(subset='Chr')
    # True labels
    y_values = data['CLASS'].values
    pre_value = data[column].values
    precision_curve1, recall_curve1, _ = precision_recall_curve(y_values, pre_value)
    auprc1 = auprc_scores[column]
    cols = column.replace('_score', '')
    cols = cols.replace('_rankscore', '')
    color = tool_color_dict.get(cols, 'skyblue')  # 默认为'skyblue'，如果没有找到对应的颜色
    plt.plot(recall_curve1, precision_curve1, label=f'{cols} ({auprc1:.3f})', color=color)
plt.plot([0, 1], [1, 0], 'k--', label='Chance')  # Add a diagonal dashed line
plt.xlabel('Recall', fontproperties=font_prop, fontsize=8)
plt.ylabel('Precision', fontproperties=font_prop, fontsize=8)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig("../result_img/img3/precision_recall_curve1.png")
plt.close()
