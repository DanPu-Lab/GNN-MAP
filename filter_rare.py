import pandas as pd


df = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/test/test1.csv',low_memory=False,index_col=0)
df1 = pd.read_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/test/delete/test2.csv',low_memory=False,index_col=0)
# 假设 df 是你的 DataFrame
# 筛选出 AF 列小于 0.01 的行
df_filtered = df[df['AF'] < 0.01]
df_filtered = df_filtered.reset_index(drop=True)
df1_filtered = df1[df1['AF'] < 0.01]
df1_filtered = df1_filtered.reset_index(drop=True)

# 查看筛选结果
df_filtered.to_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/test/test1_rare.csv')
df1_filtered.to_csv('/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/test/test2_rare.csv')
