import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    jaccard_score, matthews_corrcoef, auc, average_precision_score, precision_recall_fscore_support, \
    precision_recall_curve, roc_curve
import seaborn as sns
from torch_geometric.explain.algorithm.gnn_explainer import GNNExplainer_
from train import GATModel
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# 假设您有一个函数来加载预训练的模型
def load_model():
    # 实例化模型
    # 加载最佳模型权重
    best_model_path = '/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/best_model.pth'
    model = GATModel(test_graph.num_node_features, 32, 2, 4,
                     heads=4).to(device)

    # 加载保存的最佳模型权重
    model.load_state_dict(torch.load(best_model_path))
    return model


# 评估函数
def evaluate_model(model, dataloader):
    model.eval()

    # 不需要计算梯度
    with torch.no_grad():
        out = model(test_graph.x.to(device), test_graph.edge_index.to(device), test_graph.edge_attr.to(device))
        # 预测类别
        pred = out.argmax(dim=1)
        # 计算AUC和AUPRC
        prob = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()  # 第二类的概率
        labels = test_graph.y.cpu().numpy()
        return pred,labels,prob


print('1')
test_graph = torch.load('./data/train/val_graph.pt')
test_graph.y = torch.where(test_graph.y == -1, torch.tensor(0), test_graph.y)
# Load data
df = pd.read_csv('./data/test/test1.csv',low_memory=False)
# Dictionary to hold AUC scores
auc_scores = {}
metrics_dict = {}
# Initialize a plot
# Calculate and print AUC for each prediction column
plt.figure(figsize=(10, 8))
model = load_model()
model.to(device)
# 评估模型
y_pred, y_true, y_pred_prob = evaluate_model(model, test_graph)
df['GNN-MAP_lag'] = y_pred
df['GNN-MAP_score'] = y_pred_prob
df.to_csv('./data/test/test1ed.csv',index=False)
print('ok')
# 计算各种评估指标
fpr, tpr, thresholds = roc_curve(y_true.numpy(), y_pred_prob.numpy())
acc = accuracy_score(y_true.numpy(), y_pred.numpy())
precision = precision_score(y_true.numpy(), y_pred.numpy())
recall = recall_score(y_true.numpy(), y_pred.numpy())
f1 = f1_score(y_true.numpy(), y_pred.numpy())
tn, fp, fn, tp = confusion_matrix(y_true.numpy(), y_pred.numpy()).ravel()
specificity = tn / (tn + fp)
gmean = np.sqrt(recall * specificity)
mcc = matthews_corrcoef(y_true.numpy(), y_pred.numpy())
auc_score = roc_auc_score(y_true.numpy(), y_pred_prob.numpy())
precision_curve, recall_curve, _ = precision_recall_curve(y_true.numpy(), y_pred_prob.numpy())
auprc = auc(recall_curve, precision_curve)
# AUBPRC 没有直接的函数，但通常与AUPRC相同，取决于上下文
missing_rate = fn / (fn + tp)
recall_curve1=recall_curve
precision_curve1=precision_curve
plt.plot(recall_curve1, precision_curve1, label=f'GCN (AUC = {auprc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()  # Add a legend with the AUC scores
plt.show()  # Display the plot
metrics = {}
# Calculate each metric and store it in the dictionary
metrics['Accuracy'] = acc
metrics['Precision'], metrics['Recall'], metrics['F1 Score'] = precision,recall,f1
metrics['Specificity'] = specificity
metrics['G-Mean'] = gmean
metrics['MCC'] = mcc
metrics['Missing Rate'] = missing_rate
# Add the dictionary of metrics for the current column to the main dictionary
metrics_dict['GCN'] = metrics
print(f"{'Tool':<15}", end="")
for metric_name in metrics_dict[next(iter(metrics_dict))]:
    print(f"{metric_name:<15}", end="")
print()
for column, metrics in metrics_dict.items():
    print(f"{column:<15}", end="")
    for metric_value in metrics.values():
        print(f"{metric_value:<15.4f}", end="")
    print()
sorted_columns = sorted(auc_scores, key=auc_scores.get, reverse=True)
plt.figure(figsize=(10, 8))
# Plot ROC curve for each prediction column in sorted order
plt.plot(fpr, tpr, label=f'GCN (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--',label='Chance')  # Add a diagonal dashed line
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
model.cpu()
# 使用GNNExplainer
# 初始化一个列表来存储所有节点的特征重要性
all_node_feat_masks = []
model.eval()
explainer = GNNExplainer_(model)

# 遍历测试集中的所有样本
for data in test_list:
    node_feat_mask, edge_mask = explainer.explain_node(0, data.x.cpu(), data.edge_index.cpu())
    all_node_feat_masks.append(node_feat_mask.detach().numpy())
#node_feat_mask, edge_mask = explainer.explain_node(0, test_list[0].x.cpu(), test_list[0].edge_index.cpu())
#all_node_feat_masks.append(node_feat_mask.detach().numpy())
# 计算每个特征的重要性
all_node_feat_masks = np.array(all_node_feat_masks)

# 特征及其重要性数据
features = [
    'AF', 'AF_raw','AF_male', 'AF_female', 'AF_afr', 'AF_ami', 'AF_amr', 'AF_asj', 'AF_eas', 'AF_fin', 'AF_nfe', 'AF_oth', 'AF_sas',
    'gdi', 'gdi_phred', 'rvis1', 'rvis2', 'lof_score', 'func',
    'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL',
    'GERP++_NR','GERP++_RS','phyloP100way_vertebrate','phyloP30way_mammalian','phyloP17way_primate', 'phastCons100way_vertebrate',
    'phastCons30way_mammalian','phastCons17way_primate','GERP++_RS_rankscore','phyloP100way_vertebrate_rankscore',
    'phyloP20way_mammalian', 'phyloP20way_mammalian_rankscore','phastCons100way_vertebrate_rankscore','phastCons20way_mammalian',
    'phastCons20way_mammalian_rankscore','SiPhy_29way_logOdds','SiPhy_29way_logOdds_rankscore',
    'molecular_weight', 'equipotential_point', 'hydrophilic', 'hydrophobic', 'amphipathic', 'cyclic',
    'essential', 'aromatic', 'aliphatic', 'nonpolar', 'polar_uncharged', 'acidic', 'basic', 'sulfur', 'pka_cooh',
    'pka_nh3', 'blosum100',
    'Gm12878', 'H1hesc', 'Hepg2', 'Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf',
    'hpo0', 'hpo1', 'hpo2', 'hpo3', 'hpo4', 'hpo5', 'hpo6', 'hpo7', 'hpo8', 'hpo9'
]
categories = [
    'Allele Frequency']*13 + ['Functional Prediction']*6 + ['Splicing']*8 + ['Conservation']*17 + ['Properties']*17 + ['Biochemical']*9 + ['Phenotype']*10

# 计算每个特征的重要度均值
mean_importances = np.mean(all_node_feat_masks, axis=0)

# 创建DataFrame
feature_data = {
    'feature': features,
    'mean_importance': mean_importances,
    'category': categories
}
df_mean = pd.DataFrame(feature_data)

# 按重要性均值排序
df_mean = df_mean.sort_values(by='mean_importance', ascending=False)

# 根据排序后的特征重新组织重要度数据
sorted_features = df_mean['feature'].tolist()
sorted_categories = df_mean['category'].tolist()
sorted_importances = []

for feature in sorted_features:
    idx = features.index(feature)
    sorted_importances.append(all_node_feat_masks[:, idx])

# 创建长格式数据
feature_data = []
for i in range(len(sorted_features)):
    for importance in sorted_importances[i]:
        feature_data.append({'feature': sorted_features[i], 'importance': importance, 'category': sorted_categories[i]})

# 创建DataFrame
df = pd.DataFrame(feature_data)
print("========df==========")
print(df)
# 颜色字典
tool_color_dict = {
    'GNN-MAP': '#4C9BE6',  # 蓝色
    'REVEL': '#4D4D9F',  # 浅橙色
    'CADD': '#87CEEB',  # 绿色
    'MutPred': '#9FD1EE',  # 浅蓝色
    'MetaSVM': '#BEE8E8',  # 浅紫色
    'VEST4': '#E995C9',  # 青色
    'MetaLR': '#FC9871',  # 浅粉色
    'M-CAP': '#B0DC66',  # 灰色
    'DANN': '#75C8AE',  # 浅黄色
    'MVP': '#98A9D0'  # 浅棕色
}

# 调整后的颜色字典
palette = {
    'Allele Frequency': tool_color_dict['MutPred'],  # 浅蓝色
    'Functional Prediction': tool_color_dict['CADD'],  # 绿色
    'Splicing': tool_color_dict['GNN-MAP'],  # 红色
    'Conservation': tool_color_dict['REVEL'],  # 橙色
    'Properties': tool_color_dict['MetaSVM'],  # 紫色
    'Biochemical': tool_color_dict['M-CAP'],  # 粉色
    'Phenotype': tool_color_dict['MVP']  # 浅黄色
}

# 设置字体为Times New Roman
plt.rcParams['font.size'] = 7
# 指定字体路径
font_path = '/mnt/nfs/yuht/software/ARIAL.TTF'

# 创建FontProperties对象
font_prop = fm.FontProperties(fname=font_path)
# 将自定义字体添加到matplotlib的字体管理器中
fm.fontManager.addfont(font_path)
# 更新rcParams以使用自定义字体作为默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
# 生成Precision-Recall曲线图
plt.figure(figsize=(6.6,8.8),dpi=300)
# 设置seaborn调色板
sns.set_palette(sns.color_palette([palette[category] for category in df['category'].unique()]))

# 使用seaborn绘制箱线图
boxplot = sns.boxplot(x='importance', y='feature', hue='category', data=df,showfliers=False, linewidth=0.5, patch_artist=True, saturation=1)
# 缩短箱线图中的线
for line in boxplot.artists:
    line.set_edgecolor('black')
    for i in range(6, 12):
        line = line.get_lines()[i]
        line.set_linewidth(1)
        line.set_xdata(line.get_xdata() * 0.8)  # 缩短线的长度
plt.xlabel('Importance score', fontsize=8)
plt.ylabel('Features', fontsize=8)
# 移除图例
plt.legend([], [], frameon=False)
df.to_csv("../img/im.csv")
df_mean.to_csv("../img/mean.csv")
plt.tight_layout()  # 调整布局以避免标签被裁剪
plt.savefig('../img/importance_boxplot.pdf')
