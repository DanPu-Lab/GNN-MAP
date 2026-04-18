import numpy as np
import pandas as pd
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import IsolationForest
from sklearn.impute import IterativeImputer
new_order = ["Chr", "Start", "End", "Ref", "Alt", "CLASS", "gene","phastConsElements100way", "phyloP100way_vertebrate",
    "phyloP20way_mammalian", "phastCons100way_vertebrate", "phastCons20way_mammalian",
    "SiPhy_29way_logOdds", "phyloP30way_mammalian", "phastCons30way_mammalian", "AF", "AF_raw",
    "AF_male", "AF_female", "AF_afr", "AF_ami", "AF_amr", "AF_asj", "AF_eas", "AF_fin", "AF_nfe",
    "AF_oth", "gdi", "gdi_phred", "rvis1", "rvis2", "lof_score", "molecular_weight",
    "equipotential_point", "hydrophilic", "hydrophobic", "amphipathic ", "cyclic", "essential",
    "aromatic", "aliphatic", "nonpolar", "polar_uncharged", "acidic", "basic", "sulfur",
    "pka_cooh", "pka_nh3", "blosum100", "DS_AG", "DS_AL", "DS_DG", "DS_DL", "DP_AG", "DP_AL",
    "DP_DG", "DP_DL", "Gm12878", "H1hesc", "Hepg2", "Hmec", "Hsmm", "Huvec", "K562", "Nhek",
    "Nhlf", "func_frameshift", "func_nonframeshift", "func_nonsynonymous SNV", "func_startloss",
    "func_stopgain", "func_stoploss"
    ]
def replace_empty_with_zero(data):
    # 将空字符串和 '.' 替换为 NaN
    data.replace({'': np.nan, '.': np.nan}, inplace=True)

    # 对第21列之后的数据进行插补
    data_to_impute = data.iloc[:, 7:].copy()

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
    data_non_outliers = data.iloc[outlier_predictions == 1, :7]

    # 将清理后的前21列和插补后的数据拼接
    data_final = pd.concat([data_non_outliers.reset_index(drop=True), data_cleaned.reset_index(drop=True)], axis=1)

    return data_final
def split_dataset_by_gene(D, N, gene_column='gene'):
    """
    根据基因级别将数据集D划分为数据集A和B，以避免标签泄漏。

    参数：
    - D: pandas DataFrame，包含变异信息，必须包含基因列
    - N: float，数据集A中变异数量占D的比例阈值（0到1之间）
    - gene_column: str，表示基因列的列名，默认值为'gene'

    返回：
    - A: pandas DataFrame，数据集A，包含基因集合LA中的变异
    - B: pandas DataFrame，数据集B，包含基因集合LB中的变异
    """
    # 总的变异数量
    total_variants = len(D)

    # 获取每个基因对应的变异数量
    gene_variant_counts = D[gene_column].value_counts().to_dict()

    # 所有基因的列表LD
    LD = list(gene_variant_counts.keys())
    random.seed(42)  # 设置随机种子
    # 随机打乱基因列表
    random.shuffle(LD)

    LA = []  # 数据集A的基因列表
    LB = []  # 数据集B的基因列表
    nA = 0  # 数据集A中的变异数量
    target_nA = N * total_variants  # 数据集A的目标变异数量

    # 迭代添加基因到LA，直到nA达到目标比例
    for gene in LD:
        LA.append(gene)
        nA += gene_variant_counts[gene]
        if nA >= target_nA:
            break

    # 剩余的基因分配给LB
    LB = [gene for gene in LD if gene not in LA]

    # 构建数据集A和B
    A = D[D[gene_column].isin(LA)].reset_index(drop=True)
    B = D[D[gene_column].isin(LB)].reset_index(drop=True)

    return A, B

# 读取数据集
variants_df2 = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/predictSNP.csv.hg38_multianno.csv',low_memory=False)
variants_df1 = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/train/train_predictSNP.csv',low_memory=False)
# 假设 df1 和 df2 具有共同的键列 'key_column'
variants_df1 = variants_df1[new_order]
key_column = ['Chr', 'Start', 'End']
print(variants_df1)
print(variants_df2)
# 指定 df2 中需要保留的部分列（除了键列，还需要的其他列）
df2_columns_to_keep = ['Chr', 'Start', 'End', "ClinPred_score", "REVEL_score",
    "MetaSVM_score", "MetaLR_score", "MutPred_score"]  # 替换为你需要的列
variants_df2 = variants_df2.drop_duplicates(subset=['Chr', 'Start', 'End'])
variants_df2['Chr'] = variants_df2['Chr'].apply(lambda x: 'chr' + x if not x.startswith('chr') else x)
# 执行 left join
variants_df = pd.merge(variants_df1, variants_df2[df2_columns_to_keep], on=key_column, how='left')
print(variants_df)
variants_df = replace_empty_with_zero(variants_df)
print(variants_df)
#variants_df.to_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/test/test1.csv')
# 按照80%的比例划分数据集
A, B = split_dataset_by_gene(variants_df, N=0.9, gene_column='gene')

# 查看划分后的数据集大小
print(f"数据集A包含 {len(A)} 个变异。")
print(A['CLASS'].value_counts())
print(f"数据集B包含 {len(B)} 个变异。")
print(B['CLASS'].value_counts())
A.to_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/train/train_2.csv')
B.to_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/train/val_2.csv')

