import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager as fm
# 数据
k_values = [1,3, 5, 10, 15, 20,25,30]
auc = [0.9828,0.9919, 0.9925, 0.9923, 0.9914, 0.9907,0.9906,0.9905]
auprc = [0.9826,0.9930, 0.9938, 0.9937, 0.9928, 0.9918,0.9918,0.9917]
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

# 设置风格
sns.set_style("whitegrid")
plt.figure(figsize=(6.6,3.3),dpi=600)

# 绘制折线图
plt.plot(k_values, auc, marker='s', linestyle='--', label="AUC", color='#4C9BE6')
plt.plot(k_values, auprc, marker='^', linestyle='-.', label="AUPRC", color='#4D4D9F')

# 标注最优 k 值
# 设置标题和标签
#plt.title("Effect of k on Model Performance", fontsize=8)
plt.xlabel("k (Number of Nearest Neighbors)", fontsize=8)
plt.ylabel("Score", fontsize=8)
plt.xticks(k_values)

# **调整 Y 轴范围以放大细节**
#plt.ylim(0.957, 0.963)  # 适用于 Accuracy
plt.ylim(0.9825, 0.9945)  # 如果单独绘制 AUC & AUPRC 可以使用这个范围

# 添加图例
plt.legend(loc="lower right")

# 显示图表
plt.savefig("../result_img/feature_im/3.pdf")
plt.close()
