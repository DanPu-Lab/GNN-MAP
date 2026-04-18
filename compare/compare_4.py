import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import interpolate, stats
from scipy.interpolate import interp1d
from scipy.stats import chi2_contingency, ttest_ind, gaussian_kde
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    jaccard_score, matthews_corrcoef, auc, average_precision_score, precision_recall_fscore_support, \
    precision_recall_curve, roc_curve
import seaborn as sns
from matplotlib import font_manager as fm
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# 评估函数
def sigmoid(x, alpha=0, beta=1):
    return 1 / (1 + np.exp(-beta * (x - alpha)))

# Load data
df_patient = pd.read_csv('./delete/test3ed.csv')
df_patient['CLASS'] = df_patient['CLASS'].replace(-1, 0)
df_patient['GNN-MAP'] = df_patient['GNN-MAP_score']
y = df_patient['CLASS']  # 目标变量
y_1 = df_patient['GNN-MAP_lag']
#col_to_keep_test = col_to_keep.iloc[X.index]
# 统计正负样本数量
positive_count = np.sum(y == 0)
print(positive_count)
print(np.sum(y==1))
print("===============")
positive_count = np.sum(y_1 == 0)
print(positive_count)
print(np.sum(y_1==1))
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

def assess_variants(df_patient):
    # Update the list of genes to match those in the figure
    # 'ABCA4','USH2A','RP1L1'
    genes_of_interest = ['RP1L1']
    df_patient = df_patient[df_patient['gene'].isin(genes_of_interest)]

    # List of models to include in the analysis
    #models = ['GNN-MAP','VEST4','ClinPred']
    models = ['GNN-MAP']
    models1 = ['GNN-MAP',  # 蓝色
        'REVEL',  # 浅橙色
        'PrimateAI',  # 绿色
        'MutPred',  # 浅蓝色
        'MetaSVM',  # 浅紫色
        'VEST4',  # 青色
        'MetaLR',  # 浅粉色
        'M-CAP',  # 灰色
        'ClinPred',  # 浅黄色
        'VARITY_R']

    color1 = ['#4C9BE6','#4D4D9F','#87CEEB','#9FD1EE','#BEE8E8','#E995C9','#FC9871',
        '#B0DC66','#75C8AE','#98A9D0'
    ]

    # Initialize lists for storing model, gene, and detected pathogenic variant count
    model_list = []
    gene_list = []
    pathogenic_variant_counts = []

    # Loop through each model to calculate the number of detected pathogenic variants per gene
    for model in models1:
        threshold = prediction_thresholds[model]  # Get the threshold for the current model
        for gene in genes_of_interest:
            # Subset data for each gene and apply the threshold to the prediction scores
            subset = df_patient[
                (df_patient['gene'] == gene) & (df_patient[model] >= threshold)]  # Apply threshold to scores

            # Count the number of pathogenic variants for the gene
            count = subset.shape[0]
            model_list.append(model)
            gene_list.append(gene)
            subset2 = df_patient[(df_patient['gene']==gene) & ((df_patient[model]<threshold)==df_patient['CLASS'])]
            count1 = len(subset2)
            count2 = len(df_patient[df_patient['gene']==gene])
            acc = count1/count2
            pathogenic_variant_counts.append(count)
            # Optional: Print the result for each gene and model
            print(f"Model: {model}, Gene: {gene}, Number of Pathogenic Variants: {count},{count1},{count2},{acc}")
            high_score_threshold = 0.9
            subset1 = df_patient[(df_patient['gene'] == gene) & (df_patient['CLASS']<1) & (df_patient[model]<0.5)]
            scores = subset1[model].values
            high_score_count = np.sum(scores <0.5)
            total_count = len(scores)
            high_score_percentage = (high_score_count / total_count) * 100

            print(f"Percentage of scores >= {high_score_threshold} & {total_count}: {high_score_percentage:.2f}%")

    # Create a DataFrame for the bar plot (Figure A)
    plot_data = pd.DataFrame({
        'Model': model_list,
        'Gene': gene_list,
        'Detected Pathogenic Variants': pathogenic_variant_counts
    })
    plt.figure(figsize=(2.2, 1.65), dpi=600)
    plt.barh(model_list, pathogenic_variant_counts,color=color1)
    plt.axvline(x=85, color='black', linestyle='--', linewidth=1)

    # 设置标签和标题
    #plt.xlabel('Number of detected pathogenic variants', fontsize=8)
    plt.ylabel('Model', fontsize=8)

    # 美化图表布局
    plt.tight_layout()
    #plt.savefig("../result_img/img5/USH2A_1.pdf")
    plt.close()
    def plot_pie_in_density(scores, gene_name,t):
        # Define threshold for pathogenic vs. non-pathogenic variants
        threshold = t  # Adjust based on the model

        # Create binary labels based on the threshold
        pathogenic = scores >= threshold
        non_pathogenic = scores < threshold

        # Count pathogenic and non-pathogenic variants
        pathogenic_count = np.sum(pathogenic)
        non_pathogenic_count = np.sum(non_pathogenic)

        high_score_threshold = 0.9
        high_score_count = np.sum(scores >= high_score_threshold)
        total_count = np.sum(scores>=0.5)
        high_score_percentage = (high_score_count / total_count) * 100

        print(f"Percentage of scores >= {high_score_threshold}: {high_score_percentage:.2f}%")
        # Create the main figure for score distribution
        fig, ax = plt.subplots(figsize=(2.2, 1.65), dpi = 600)

        # Plot density (score distribution) using seaborn
        sns.kdeplot(scores, fill=True, color='#75C8AE', ax=ax, alpha=0.6)
        ax.axvline(x=threshold, color='black', linestyle='--', linewidth=1)  # Vertical line at threshold
        ax.set_xlim(0, 1)
        ax.set_ylabel('', fontsize=8)
        # Add inset pie chart inside the density plot
        ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper right', borderpad=2)
        custom_colors = ['#75C8AE','#98A9D0']
        ax_inset.pie([pathogenic_count, non_pathogenic_count], startangle=90,colors=custom_colors)

        # Adjust layout and show plot
        plt.tight_layout()
        #plt.savefig("../result_img/img5/RP1L1_ClinPred.pdf")
        plt.close()
    # Figure B: Distribution of pathogenicity scores for each model and gene
    for gene in genes_of_interest:
        for i, model in enumerate(models):
            threshold = prediction_thresholds[model]
            subset = df_patient[(df_patient['gene'] == gene) & (df_patient['CLASS']>=1)]
            score_data = subset[
                model].values  # Assuming the model's prediction score is stored in the column with model name
            plot_pie_in_density(score_data,'USH2A',threshold)


ax2 = 1
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
# 加载数据
df = pd.read_csv('./delete/test3ed.csv')
df['CLASS'] = df['CLASS'].replace(-1, 0)
num = df['AF']
num_sum = np.sum(num == 0)
df['GNN-MAP'] = df['GNN-MAP_score']
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

fig, ax5 = plt.subplots(figsize=(3.3,3.3), dpi=100)
assess_variants(df_patient)
