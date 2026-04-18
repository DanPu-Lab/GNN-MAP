import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt, gridspec, font_manager
from matplotlib.font_manager import FontProperties
from scipy import interpolate
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    jaccard_score, matthews_corrcoef, auc, average_precision_score, precision_recall_fscore_support, \
    precision_recall_curve, roc_curve
import matplotlib.font_manager as fm

tool_color_dict = {
    'GNN-MAP': '#4C9BE6',  # 蓝色
    'REVEL': '#4D4D9F',  # 浅橙色
    'CAPICE': '#75C8AE',  # 绿色
    'MutPred': '#9FD1EE',  # 浅蓝色
    'MetaSVM': '#BEE8E8',  # 浅紫色
    'VEST4': '#E995C9',  # 青色
    'MetaLR': '#FC9871',  # 浅粉色
    'M-CAP': '#B0DC66',  # 灰色
    'DANN': '#75C8AE',  # 浅黄色
    'MVP': '#98A9D0'  # 浅棕色
}

# 调整后的颜色列表
colors = ['#FBCB1F', '#F2F3C7']
colors1 = ['#1C6AB1', '#0D95CE', '#5DBDE8', '#89BCE4']

df = pd.read_csv('./delete/test2ed_rare.csv')
df1 = pd.read_csv('./test1ed_rare.csv')
num = df['AF']
num_sum = np.sum(num == 0)
df['CLASS'] = df['CLASS'].replace(-1, 0)
df['GNN-MAP'] = df['GNN-MAP_score']
df1['CLASS'] = df1['CLASS'].replace(-1, 0)
df1['GNN-MAP'] = df1['GNN-MAP_score']
# 列出预测列
predprod_columns = ['M-CAP','REVEL','PrimateAI',
    'GNN-MAP']
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

def fun(limit_min,limit_max,flag1,flag2,df):
    # Dictionary to hold AUC scores
    auc_scores1 = {}
    auprc_scores1 = {}
    for column in predprod_columns:
        df[column] = df[column].fillna(df[column].mean())
        data = df.dropna(subset=column)
        data = data[data['AF'] < limit_max].dropna(subset=column)
        data = data[data['AF'] > limit_min].dropna(subset=column)
        data = data.dropna(subset='Chr')
        print(data['CLASS'].value_counts())
        # True labels
        y_values = data['CLASS'].values
        pred_prob = data[column].values  # Extract the prediction values for the current column
        #pred_prob = sigmoid(pred_prob, alpha=1.5, beta=1)
        try:
            auc_score1 = roc_auc_score(y_values, pred_prob)
            precision_curve1, recall_curve1, _ = precision_recall_curve(y_values, pred_prob)
            auprc1 = auc(recall_curve1, precision_curve1)
            auprc_scores1[column] = auprc1
            auc_scores1[column] = auc_score1
            print(f"========{column},{limit_min}==========")
            print(auprc1,auc_score1)
        except ValueError as e:
            # Handle cases where AUC cannot be computed
            print(f'Could not compute AUC for {column}: {e}')
    auc_scores1['CAPICE'] = 0.979
    auprc_scores1['CAPICE'] = 0.976
    sorted_columns2 = sorted(auc_scores1, key=auc_scores1.get, reverse=True)
    sorted_columns3 = sorted(auprc_scores1, key=auprc_scores1.get, reverse=True)
    # Plot ROC curve for each prediction column in sorted order
    fig, ax7 = plt.subplots(figsize=(2.2, 1.65), dpi=600)
    for column in sorted_columns2:
        if column != 'CAPICE':
            data = df.dropna(subset=column)
            data = data.dropna(subset='Chr')
            # True labels
            y_values = data['CLASS'].values
            pre_value = data[column].values
            fpr, tpr, thresholds = roc_curve(y_values, pre_value)
        auc_score = auc_scores1[column]
        tool_name = column.replace('_rankscore', '')
        tool_name = tool_name.replace('_score', '')
        color = tool_color_dict.get(tool_name, 'skyblue')  # 默认为'skyblue'，如果没有找到对应的颜色
        ax7.barh(tool_name, auc_score, color=color)
    ax7.set_xlabel(f'AUC (AF<1.0%)', fontsize=8)
    ax7.set_ylabel('Tool', fontsize=8)
    #ax7.text(-0.4, 1.08, flag1, transform=ax7.transAxes,fontsize=20, va='top', ha='right')
    plt.tight_layout()
    plt.savefig(f"../result_img/img4/{flag1}.png")
    plt.close()
    fig, ax8 = plt.subplots(figsize=(2.2, 1.65), dpi=600)
    for column in sorted_columns3:
        if column != 'CAPICE':
            data = df.dropna(subset=column)
            data = data.dropna(subset='Chr')
            # True labels
            y_values = data['CLASS'].values
            pre_value = data[column].values
            precision_curve1, recall_curve1, _ = precision_recall_curve(y_values, pre_value)
        auprc1 = auprc_scores1[column]
        tool_name = column.replace('_rankscore', '')
        tool_name = tool_name.replace('_score', '')
        color = tool_color_dict.get(tool_name, 'skyblue')  # 默认为'skyblue'，如果没有找到对应的颜色
        ax8.barh(tool_name, auprc1, color=color)
    ax8.set_xlabel(f'AUPRC (AF<1.0%)', fontsize=8)
    ax8.set_ylabel('Tool', fontsize=8)
    #ax8.text(-0.4, 1.08, flag2, transform=ax8.transAxes, fontsize=20, va='top', ha='right')
    plt.tight_layout()
    plt.savefig(f"../result_img/img4/{flag2}.png")
    plt.close()

# 创建一个包含3个主子图的图像
# 设置字体为Arial
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
#fun(limit_min=0,limit_max=0.01,flag1='A',flag2='B',df=df)
fun(limit_min=0,limit_max=0.01,flag1='C1',flag2='D1',df =df1)
