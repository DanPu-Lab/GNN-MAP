import warnings
import joblib
from imblearn.over_sampling import SMOTE
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
import torch
from tqdm import tqdm
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/train/train.csv',low_memory=False,index_col=0)  # 示例：从CSV文件中读取数据
df1 = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/train/val.csv',low_memory=False,index_col=0)  # 示例：从CSV文件中读取数据


# 计算PCC（皮尔逊相关系数）
def compute_pearson_similarity(features):
    if np.any(np.std(features, axis=0) == 0):
        warnings.warn("One or more features have zero variance.")
    correlation_matrix = np.corrcoef(features.T, rowvar=False)
    np.nan_to_num(correlation_matrix, copy=False)  # 将NaN替换为0
    similarity_matrix = (correlation_matrix + 1) / 2  # 将PCC值映射到[0, 1]之间
    return similarity_matrix

# 基于KNN构建图的边
def compute_knn_edges(features, similarity_threshold=0.95, k=0):
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
scaler = MinMaxScaler(feature_range=(0, 1))
col = ['Chr','Start','End','Ref','Alt','CLASS','gene',
       'omim_Autosomal_dominant','omim_Autosomal_recessive',
       'omim_X_linked_dominant','omim_X_linked_recessive',
       'omim_other','func_frameshift', 'func_nonframeshift', 'func_nonsynonymous SNV',
    'func_startloss', 'func_stopgain', 'func_stoploss']
col2 = ['Chr','Start','End','Ref','Alt','CLASS','gene','VARITY_R_score',
       'VEST4_score','M-CAP_score','PrimateAI_score','MutationAssessor_score',
       'LIST-S2_score','SIFT4G_score','DANN_rankscore','MutationTaster_score',
        'func_frameshift', 'func_nonframeshift', 'func_nonsynonymous SNV',
    'func_startloss', 'func_stopgain', 'func_stoploss'
]
df = df.reset_index(drop=True)
df1 = df1.reset_index(drop=True)
X_train = df.drop(col, axis=1)
X = df.drop(col, axis=1)
X_test = df1.drop(col, axis=1)
y_train = df['CLASS']
y = df['CLASS']
y_test = df1['CLASS']

X.columns = X.columns.astype(str)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, './process/scaler_k1.gz')
print("ok")

df_resampled = pd.DataFrame(X_train, columns=X.columns)
df1_resampled = pd.DataFrame(X_test, columns=X.columns)
target_df = pd.DataFrame(y_train, columns=['CLASS'])
target1_df = pd.DataFrame(y_test, columns=['CLASS'])

df_train = pd.concat([df_resampled.reset_index(drop=True), target_df.reset_index(drop=True)], axis=1)
df_test = pd.concat([df1_resampled, target1_df], axis=1)
print(df_train.columns)
print("=============================================")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def create(df, k=0, similarity_threshold=0.95):
    # 获取节点特征
    feature1 = df.iloc[:, :22]
    feature2 = df.iloc[:, 24:42]
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
data2 = create(df_test)
# 保存训练图和测试图
torch.save(data1, './data/train/train_graph_k0.pt')
torch.save(data2, './data/train/val_graph_k0.pt')

print("图数据保存成功。")