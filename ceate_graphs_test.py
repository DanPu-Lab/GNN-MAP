import warnings
import joblib
from sklearn.experimental import enable_iterative_imputer
from imblearn.over_sampling import SMOTE
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import IsolationForest
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
import torch
from tqdm import tqdm
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors

#df = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/test3.csv',low_memory=False)  # 示例：从CSV文件中读取数据
df = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/test/test2.csv',low_memory=False)  # 示例：从CSV文件中读取数据


def replace_empty_with_zero(data):
    # 将空字符串和 '.' 替换为 NaN
    data.replace({'': np.nan, '.': np.nan}, inplace=True)

    # 对第21列之后的数据进行插补
    data_to_impute = data.iloc[:, -5:].copy()

    # 使用 IterativeImputer 进行缺失值插补
    imputer = IterativeImputer(max_iter=10, random_state=0)
    df_imputed = imputer.fit_transform(data_to_impute)

    # 将插补结果转为 DataFrame
    data_imputed = pd.DataFrame(df_imputed, columns=data_to_impute.columns)

    # 使用 IsolationForest 检测并移除离群值
    iso_forest = IsolationForest(contamination=0.05)  # 设置离群点的比例
    outlier_predictions = iso_forest.fit_predict(data_imputed)

    # 筛选出未被标记为离群值的行
    data_cleaned = data_imputed[outlier_predictions == 1]

    # 同步原始数据前21列的行，删除对应的离群值行
    data_non_outliers = data.iloc[outlier_predictions == 1, :-5]

    # 将清理后的前21列和插补后的数据拼接
    data_final = pd.concat([data_non_outliers.reset_index(drop=True), data_cleaned.reset_index(drop=True)], axis=1)

    return data_final
# 计算PCC（皮尔逊相关系数）
def compute_pearson_similarity(features):
    if np.any(np.std(features, axis=0) == 0):
        warnings.warn("One or more features have zero variance.")
    correlation_matrix = np.corrcoef(features.T, rowvar=False)
    np.nan_to_num(correlation_matrix, copy=False)  # 将NaN替换为0
    similarity_matrix = (correlation_matrix + 1) / 2  # 将PCC值映射到[0, 1]之间
    return similarity_matrix

# 基于KNN构建图的边
def compute_knn_edges(features, similarity_threshold=0.95, k=5):
    # 计算PCC相似性矩阵
    similarity_matrix = compute_pearson_similarity(features)

    # 使用KNN选择邻居
    nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed')
    nbrs.fit(1 - similarity_matrix)  # 使用1-PCC作为“距离”
    distances, indices = nbrs.kneighbors(1 - similarity_matrix)

    edge_index = []
    edge_weights = []

    for i in range(similarity_matrix.shape[0]):
        neighbor_similarities = similarity_matrix[i, indices[i]]

        # 根据相似性阈值过滤邻居
        valid_mask = neighbor_similarities > similarity_threshold
        valid_indices = indices[i][valid_mask]
        valid_similarities = neighbor_similarities[valid_mask]

        for j, neighbor_index in enumerate(valid_indices):
            # 添加边并记录权重
            edge_index.append([i, neighbor_index])
            edge_index.append([neighbor_index, i])  # 无向图
            edge_weights.append(valid_similarities[j])
            edge_weights.append(valid_similarities[j])

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), torch.tensor(edge_weights, dtype=torch.float)
# 使用KNN选择邻居
print(df['CLASS'].value_counts())
scaler = joblib.load('./process/scaler.gz')  # 加载scaler
col2 = ['CLASS']
columns_list2 = [
    "Chr", "Start", "End", "Ref", "Alt", "CLASS", "gene","phastConsElements100way",
    "phyloP100way_vertebrate", "phyloP20way_mammalian", "phastCons100way_vertebrate",
    "phastCons20way_mammalian", "SiPhy_29way_logOdds", "phyloP30way_mammalian",
    "phastCons30way_mammalian", "AF", "AF_raw", "AF_male", "AF_female", "AF_afr", "AF_ami",
    "AF_amr", "AF_asj", "AF_eas", "AF_fin", "AF_nfe", "AF_oth", "gdi", "gdi_phred", "rvis1",
    "rvis2", "lof_score", "molecular_weight", "equipotential_point", "hydrophilic",
    "hydrophobic", "amphipathic ", "cyclic", "essential", "aromatic", "aliphatic",
    "nonpolar", "polar_uncharged", "acidic", "basic", "sulfur", "pka_cooh", "pka_nh3",
    "blosum100", "DS_AG", "DS_AL", "DS_DG", "DS_DL", "DP_AG", "DP_AL", "DP_DG", "DP_DL",
    "Gm12878", "H1hesc", "Hepg2", "Hmec", "Hsmm", "Huvec", "K562", "Nhek", "Nhlf",
    "func_frameshift", "func_nonframeshift", "func_nonsynonymous SNV", "func_startloss",
    "func_stopgain", "func_stoploss", "omim_Autosomal_dominant", "omim_Autosomal_recessive",
    "omim_X_linked_dominant", "omim_X_linked_recessive", "omim_other", "ClinPred_score",
    "REVEL_score", "MetaSVM_score", "MetaLR_score", "MutPred_score"
]
columns_list3 = [
    "Chr", "Start", "End", "Ref", "Alt", "CLASS", "gene", "phastConsElements100way",
    "phyloP100way_vertebrate", "phyloP20way_mammalian", "phastCons100way_vertebrate",
    "phastCons20way_mammalian", "SiPhy_29way_logOdds", "phyloP30way_mammalian",
    "phastCons30way_mammalian", "AF", "AF_raw", "AF_male", "AF_female", "AF_afr", "AF_ami",
    "AF_amr", "AF_asj", "AF_eas", "AF_fin", "AF_nfe", "AF_oth", "gdi", "gdi_phred", "rvis1",
    "rvis2", "lof_score", "molecular_weight", "equipotential_point", "hydrophilic",
    "hydrophobic", "amphipathic ", "cyclic", "essential", "aromatic", "aliphatic",
    "nonpolar", "polar_uncharged", "acidic", "basic", "sulfur", "pka_cooh", "pka_nh3",
    "blosum100", "DS_AG", "DS_AL", "DS_DG", "DS_DL", "DP_AG", "DP_AL", "DP_DG", "DP_DL",
    "Gm12878", "H1hesc", "Hepg2", "Hmec", "Hsmm", "Huvec", "K562", "Nhek", "Nhlf",
    "func_frameshift", "func_nonframeshift", "func_nonsynonymous SNV", "func_startloss",
    "func_stopgain", "func_stoploss", "ClinPred_score",
    "REVEL_score", "MetaSVM_score", "MetaLR_score", "MutPred_score"
]
new = ['CLASS', 'phyloP100way_vertebrate',
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
       "ClinPred_score", "REVEL_score","MetaSVM_score", "MetaLR_score", "MutPred_score"]
df = df[columns_list3]
#df = replace_empty_with_zero(df)
#df.to_csv('./data/test/test3.csv')
df = df[new]
X_train = df.drop(col2, axis=1)
X = df.drop(col2, axis=1)
y_train = df['CLASS']
y = df['CLASS']

X.columns = X.columns.astype(str)
X_train = scaler.transform(X_train)

print("ok")

df_resampled = pd.DataFrame(X_train, columns=X.columns)
target_df = pd.DataFrame(y_train, columns=['CLASS'])

df_train = pd.concat([df_resampled, target_df], axis=1)
print(df_train.columns)
print("=============================================")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def create(df, k=5, similarity_threshold=0.95):
    # 获取节点特征42
    feature1 = df.iloc[:, :21]
    feature2 = df.iloc[:, 23:41]
    feature = df.iloc[:, :-1].values
    features = pd.concat([feature1, feature2], axis=1).values

    # 计算边和权重
    edge_index, edge_weights = compute_knn_edges(features, similarity_threshold, k)
    # 将特征转换为PyTorch张量
    node_features = torch.tensor(feature, dtype=torch.float)
    # 获取标签
    labels = torch.tensor(df['CLASS'].values, dtype=torch.long)
    # 构建图数据对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights, y=labels)
    return data

data1 = create(df_train)
torch.save(data1, './data/test/test2_final_graph.pt')

print("图数据保存成功。")