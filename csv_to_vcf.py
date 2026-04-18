import pandas as pd

# 读取 CSV 文件
csv_file = "/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/test/test1ed.csv"  # 输入 CSV 文件路径
#csv_file = "/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/compare/delete/test3ed.csv"  # 输入 CSV 文件路径
df = pd.read_csv(csv_file)  # 假设 CSV 以制表符（Tab）分隔

# VCF 文件的头部信息
vcf_header = """##fileformat=VCFv4.0
##reference=GRCh38
#CHROM\tPOS\tID\tREF\tALT
"""

# 输出 VCF 文件路径
vcf_file = "/mnt/nfs/yuht/test/pythonProject1/GNN-MAP/data/test/subinput/test1ed.vcf"

# 打开 VCF 文件进行写入
with open(vcf_file, 'w') as vcf:
    vcf.write(vcf_header)  # 写入头部信息

    # 遍历 CSV 数据，并格式化为 VCF 结构
    for index, row in df.iterrows():
        chrom = row['Chr']
        pos = row['Start']  # VCF 只需要 Start 作为 POS
        ref = row['Ref']
        alt = row['Alt']
        vcf.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\n")

print(f"VCF file has been successfully created: {vcf_file}")

