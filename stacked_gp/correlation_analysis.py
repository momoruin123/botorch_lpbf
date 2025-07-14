import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # 或者 "Qt5Agg"、"Agg"
# 读取 Excel 文件（改成你的文件路径）
df = pd.read_excel("fused_results_corrected2_with objective and batch4_Tensile test.xlsx")

# 只保留数值型列
numeric_df = df.select_dtypes(include='number')

# 计算皮尔逊相关系数
corr_matrix = numeric_df.corr(method='pearson')

# 绘图
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of All Numeric Features")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
