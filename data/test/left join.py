# 读取数据集
import numpy as np
import pandas as pd

variants_df2 = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/test2.csv',low_memory=False)
variants_df1 = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/test/delete/test2ed_rare.csv',low_memory=False,index_col=0)
# 假设 df1 和 df2 具有共同的键列 'key_column'
key_column = ['Chr', 'Start', 'End']
#variants_df2['VARITY_R_score'] = variants_df2['VARITY_R']
predprod_columns = ['MetaSVM_score', 'MetaLR_score', 'M-CAP_score', 'VARITY_R_score',
                    'MutPred_score', 'ClinPred_score', 'VEST4_score', 'REVEL_score',
                    'PrimateAI_score']
# 假设 df 是你需要修改的 DataFrame
# 将含有 "_score" 后缀的列名去掉后缀
new_columns = [col.replace('_score', '') for col in predprod_columns]
# 然后将这些列在 df 中重命名
variants_df2.rename(columns=dict(zip(predprod_columns, new_columns)), inplace=True)
# 指定 df2 中需要保留的部分列（除了键列，还需要的其他列）
df2_columns_to_keep = ['Chr', 'Start', 'End', 'MetaSVM', 'MetaLR', 'M-CAP', 'VARITY_R',
                    'MutPred', 'ClinPred', 'VEST4', 'REVEL',
                    'PrimateAI']  # 替换为你需要的列
variants_df2['Chr'] = variants_df2['Chr'].apply(lambda x: 'chr' + x if not x.startswith('chr') else x)
# 执行 left join
print(len(variants_df1))
print(len(variants_df2))
# 去除 variants_df2 中 key_column 列的重复项，只保留第一项或根据某列进行去重
variants_df2_unique = variants_df2.drop_duplicates(subset=key_column)

# 然后进行 merge
variants_df = pd.merge(variants_df1, variants_df2_unique[df2_columns_to_keep], on=key_column, how='left')
print(len(variants_df))
variants_df.replace({'': np.nan, '.': np.nan}, inplace=True)
variants_df.to_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/compare/delete/test2ed_rare.csv')