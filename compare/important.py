import numpy as np
import pandas as pd
import torch
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge
import seaborn as sns
from torch_geometric.explain import GNNExplainer
from torch_geometric.explain.algorithm.gnn_explainer import GNNExplainer_
from tqdm import tqdm


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1, dropout=0.5):
        super(GATModel, self).__init__()

        # 初始化图注意力层 (GATConv)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))

        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))

        # 跳跃连接 (Jumping Knowledge, JK)
        self.jk = JumpingKnowledge(mode='cat')

        # 全连接层，最后输出分类结果
        self.fc = torch.nn.Linear(num_layers * hidden_channels * heads, out_channels)

        # Dropout for regularization
        self.dropout = dropout

        # 残差连接层：用于调整维度
        self.residual_connections = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:  # 第一层的输入与隐藏层的维度不一定相同
                self.residual_connections.append(torch.nn.Linear(in_channels, hidden_channels * heads))
            else:
                self.residual_connections.append(torch.nn.Identity())  # 如果维度匹配，直接使用Identity

    def forward(self, x, edge_index, edge_attr=None):
        # 存储每层的节点嵌入
        layer_outs = []

        # 逐层图注意力卷积
        for i, conv in enumerate(self.convs):
            identity = x  # 保存残差连接的输入

            # 图卷积操作
            x = F.elu(conv(x, edge_index, edge_attr))

            # 残差连接：调整输入维度或直接相加
            if isinstance(self.residual_connections[i], torch.nn.Linear):
                identity = self.residual_connections[i](identity)
            x = x + identity  # 残差连接

            # 存储每一层的输出
            layer_outs.append(x)

            # Dropout 正则化
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 使用跳跃连接 (Jumping Knowledge)
        x = self.jk(layer_outs)

        return self.fc(x)

test_graph = torch.load('../data/test/test1_graph.pt')
test_graph.y = torch.where(test_graph.y == -1, torch.tensor(0), test_graph.y)
best_model_path = '/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/best_model.pth'
model = GATModel(test_graph.num_node_features, 32, 2, 4,heads=4)
# 加载保存的最佳模型权重
model.load_state_dict(torch.load(best_model_path))

all_node_feat_masks = []
model.eval()
explainer = GNNExplainer_(model)

# 将图和模型迁移到 GPU（如果有）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_graph = test_graph.to(device)
model = model.to(device)
# 使用 tqdm 包装 for 循环来显示进度条
# 使用 tqdm 包装 for 循环来显示进度条
for node_idx in tqdm(range(test_graph.x.size(0)), desc="Explaining nodes"):
    # 不要再调用 .cpu()，让张量保持在 GPU 上
    node_feat_mask, edge_mask = explainer.explain_node(
        node_idx,
        test_graph.x,  # 确保 x 和 edge_index 都在 GPU 上
        test_graph.edge_index
    )
    # 将结果迁移回 CPU 并转为 NumPy 数组
    all_node_feat_masks.append(node_feat_mask.detach().cpu().numpy())
#node_feat_mask, edge_mask = explainer.explain_node(0, test_list[0].x.cpu(), test_list[0].edge_index.cpu())
#all_node_feat_masks.append(node_feat_mask.detach().numpy())
# 计算每个特征的重要性
all_node_feat_masks = np.array(all_node_feat_masks)

# 特征及其重要性数据
features = [
    'phastConsElements100way', 'phyloP100way_vertebrate',
       'phyloP20way_mammalian', 'phastCons100way_vertebrate',
       'phastCons20way_mammalian', 'SiPhy_29way_logOdds',
       'phyloP30way_mammalian', 'phastCons30way_mammalian', 'AF', 'AF_raw',
       'AF_male', 'AF_female', 'AF_afr', 'AF_ami', 'AF_amr', 'AF_asj',
       'AF_eas', 'AF_fin', 'AF_nfe', 'AF_oth', 'gdi', 'gdi_phred', 'rvis1',
       'rvis2', 'lof_score', 'molecular_weight', 'equipotential_point',
       'hydrophilic', 'hydrophobic', 'amphipathic ', 'cyclic', 'essential',
       'aromatic', 'aliphatic', 'nonpolar', 'polar_uncharged', 'acidic',
       'basic', 'sulfur', 'pka_cooh', 'pka_nh3', 'blosum100', 'DS_AG', 'DS_AL',
       'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL', 'Gm12878',
       'H1hesc', 'Hepg2', 'Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf',
       "ClinPred_score", "REVEL_score","MetaSVM_score", "MetaLR_score", "MutPred_score"
]
categories = ['Conservation']*8 + [
    'Allele Frequency']*12 + ['Functional Prediction']*5+ ['Properties']*17 + ['Splicing']*8 + ['Biochemical']*9+ ['Other tool']*5

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
    'Phenotype': tool_color_dict['MVP'],  # 浅黄色
    'Other tool': tool_color_dict['CADD']
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
