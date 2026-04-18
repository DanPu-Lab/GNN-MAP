import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy import interpolate, stats
from scipy.interpolate import interp1d
from scipy.stats import chi2_contingency, ttest_ind, gaussian_kde
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    jaccard_score, matthews_corrcoef, auc, average_precision_score, precision_recall_fscore_support, \
    precision_recall_curve, roc_curve
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from torch_geometric.explain import GNNExplainer
from torch_geometric.explain.algorithm.gnn_explainer import GNNExplainer_
from matplotlib import font_manager as fm
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# 评估函数
def sigmoid(x, alpha=0, beta=1):
    return 1 / (1 + np.exp(-beta * (x - alpha)))

# Load data
df_patient = pd.read_csv('./test1ed.csv')
df_patient['CLASS'] = df_patient['CLASS'].replace(-1, 0)
df_patient['GNN-MAP'] = df_patient['GNN-MAP_score']
df_patient['Missence'] = df_patient['func_nonsynonymous SNV']
df_patient['Stopgain'] = df_patient['func_stopgain']
df_patient['Frameshift'] = df_patient['func_frameshift']
df_patient['Startloss'] = df_patient['func_startloss']
df_patient['Nonframeshift'] = df_patient['func_nonframeshift']
df_patient['Stoploss'] = df_patient['func_stoploss']
# List of prediction columns
predprod_columns = ['MetaSVM','MetaLR','M-CAP','VARITY_R',
    'MutPred','ClinPred','VEST4','REVEL','PrimateAI',
    'GNN-MAP']
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
# 加载数据
df = pd.read_csv('./test1ed.csv')
df['CLASS'] = df['CLASS'].replace(-1, 0)
num = df['AF']
num_sum = np.sum(num == 0)
df['GNN-MAP'] = df['GNN-MAP_score']
# 列出预测列
predprod_columns1 = ['MetaSVM','MetaLR','M-CAP','VARITY_R',
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
for column in predprod_columns1:
    data = df.dropna(subset=[column, 'Chr'])
    # True labels
    y_values = data['CLASS'].values
    pre_value = data[column].values  # Extract the prediction values for the current column
    optimal_threshold = prediction_thresholds[column]
    print('===')
    print(optimal_threshold)

    # 转换 CLINSIG 列中的值
    pre_value = (pre_value > optimal_threshold).astype(int)

    try:
        metrics = {}
        # Calculate each metric and store it in the dictionary
        metrics['Accuracy'] = accuracy_score(y_values, pre_value)
        metrics['Precision'], metrics['Recall'], metrics['F1 Score'], _ = precision_recall_fscore_support(y_values,
                                                                                                          pre_value,
                                                                                                          average='binary')
        tn, fp, fn, tp = confusion_matrix(y_values, pre_value).ravel()
        metrics['Specificity'] = tn / (tn + fp)
        metrics['G-Mean'] = np.sqrt(metrics['Recall'] * metrics['Specificity'])
        metrics['MCC'] = matthews_corrcoef(y_values, pre_value)
        num_all = df.shape[0]
        print(num_all)
        num_notnull = sum(df[column].isnull())
        print(num_notnull)
        metrics['Missing Rate'] = num_notnull/num_all
        print(metrics['Missing Rate'])
        # Add the dictionary of metrics for the current column to the main dictionary
        metrics_dict[column] = metrics
    except ValueError as e:
        # Handle cases where AUC cannot be computed
        print(f'Could not compute metrics for {column}: {e}')

# 按错误率排序
sorted_metrics = sorted(metrics_dict.items(), key=lambda item: item[1]['Missing Rate'])
def assess4(df_patient):
    # Define mapping from numeric to string variant types
    variant_columns = ['Missence', 'Stopgain', 'Frameshift', 'Startloss', 'Nonframeshift','Stoploss']
    tool_names = []
    variant_types = []
    non_empty_ratios = []

    for i in range(10):
        tool_name = predprod_columns[i].replace('_score', '')  # Assuming predprod_columns contains the tool names
        df_patient['Variant_Type_Label'] = df_patient[variant_columns].idxmax(axis=1)

        for variant_type in df_patient['Variant_Type_Label'].unique():
            subset = df_patient[df_patient['Variant_Type_Label'] == variant_type]
            non_empty_ratio = subset[predprod_columns[i]].notna().mean()

            # Append the data to the lists
            tool_names.append(tool_name)
            variant_types.append(variant_type)
            non_empty_ratios.append(non_empty_ratio)

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Tool': tool_names,
        'Variant Type': variant_types,
        'Non-Empty Ratio': non_empty_ratios
    })

    # Map the Variant Type to numeric values for jittering
    plot_data['Variant Type Numeric'] = plot_data['Variant Type'].map({v: i for i, v in enumerate(variant_columns)})

    # Add jitter to x-axis to prevent overlap
    plot_data['Jittered Variant Type'] = plot_data['Variant Type Numeric'] + np.random.uniform(-0.3, 0.3, plot_data.shape[0])
    palette = [tool_color_dict[tool] for tool in plot_data['Tool'].unique()]
    # Create the scatter plot with jittered x-axis
    sns.scatterplot(ax=ax2, data=plot_data, x='Jittered Variant Type', y='Non-Empty Ratio', hue='Tool', style='Tool', s=56, palette=palette,marker='o')

    # Set X-axis to show original variant types without jitter
    ax2.set_xticks(ticks=range(len(variant_columns)), labels=variant_columns)
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
    # Set Y-axis limits and ticks
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.set_ylabel('Percentage of predicted variants', fontsize=8)
    ax2.set_xlabel('Variant type', fontsize=8)
    #ax2.text(-0.2, 1.15, 'B', transform=ax2.transAxes, fontsize=18,  va='top',
             #ha='right')
    legend = ax2.legend(frameon=False,loc='upper center', bbox_to_anchor=(0.4, -0.25), ncol=3)
    # 调整图例中图形的大小
    for handle in legend.legendHandles:
        handle.set_markersize(8)  # 调整标记大小


def assess5(df_patient):
    # Define mapping from numeric to string variant types
    variant_columns = ['Missence', 'Stopgain', 'Frameshift', 'Startloss', 'Nonframeshift', 'Stoploss']
    df_patient['func_label'] = df_patient[variant_columns].idxmax(axis=1)

    precision_list = []
    func_list = []

    for func in df_patient['func_label'].unique():
        subset = df_patient[df_patient['func_label'] == func]
        pred = subset['GNN-MAP_lag'].values
        y = subset['CLASS'].values
        Precision, Recall, F1Score, _ = precision_recall_fscore_support(y, pred, average='binary')
        precision_list.append(Precision)
        func_list.append(func)
        print(Precision)

    # Create a DataFrame with the extracted data
    plot_data = pd.DataFrame({
        'Variant Type': func_list,
        'Precision': precision_list
    })
    sns.barplot(x='Variant Type', y='Precision', data=plot_data, palette=['#4C9BE6'], ax=ax3)

    # Set Y-axis limits and ticks
    ax3.set_ylim(0, 1.1)
    ax3.set_yticks(np.arange(0, 1.1, 0.1))

    # Set X-axis labels to match the example
    #ax3.set_xticklabels(ax3.get_xticklabels())
    plt.setp(ax3.get_xticklabels(), rotation=20, ha="right")
    ax3.set_ylabel('Precision', fontsize=8)
    ax3.set_xlabel('Variant type', fontsize=8)

    #ax3.text(-0.2, 1.15, 'C', transform=ax3.transAxes, fontproperties=font_prop, fontsize=18, va='top',
             #ha='right')

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
fig, ax1 = plt.subplots(figsize=(2.2,1.65), dpi=100)
# 转换 CLINSIG 列中的值
for column, metrics in sorted_metrics:
    tool_name = column.replace('_rankscore', '')
    tool_name = tool_name.replace('_score', '')
    color = tool_color_dict.get(tool_name, 'skyblue')  # 默认为'skyblue'，如果没有找到对应的颜色
    print(tool_name)
    print(metrics['Missing Rate'])
    ax1.barh(tool_name, metrics['Missing Rate'], color=color)
ax1.set_ylabel('Tool', fontsize=8)
ax1.set_xlabel('Missing rate', fontsize=8)
#ax1.text(-0.2, 1.08, 'A', transform=ax1.transAxes, fontsize=10, va='top', ha='right')
ax1.set_xlim(0, 1)  # 错误率范围在0到0.5之间
plt.tight_layout()
#plt.savefig("../result_img/img4/4.pdf")
plt.close()
fig, ax2 = plt.subplots(figsize=(2.2,3.3), dpi=500)
assess4(df_patient)
plt.tight_layout()
#plt.savefig("../result_img/img4/2.pdf")
#plt.show()
plt.close()
fig, ax3 = plt.subplots(figsize=(2.2,1.65), dpi=100)
assess5(df_patient)
plt.tight_layout()
#plt.savefig("../result_img/img4/6.pdf")
plt.close()
