import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal

# 设置两组数据（健康人 vs 病人）的参数
# 健康人 (蓝色山峰): 平均值在 (0, 0)，比较集中
mu_healthy = [0, 0]
cov_healthy = [[1, 0.3], [0.3, 1]]  # 协方差矩阵

# 病人 (红色山峰): 平均值在 (2.5, 2.5)，且分布更散一些
mu_sick = [2.5, 2.5]
cov_sick = [[1.5, 0], [0, 1.5]]

# 创建网格数据 (X, Y 轴)
x = np.linspace(-3, 6, 100)
y = np.linspace(-3, 6, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# 计算 Z 轴高度 (概率密度 PDF)
Z_healthy = multivariate_normal(mu_healthy, cov_healthy).pdf(pos)
Z_sick = multivariate_normal(mu_sick, cov_sick).pdf(pos)

# 开始绘图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 1. 绘制蓝色山峰 (健康)
surf1 = ax.plot_surface(X, Y, Z_healthy, cmap='Blues', alpha=0.7, linewidth=0.2, edgecolors='blue')

# 2. 绘制红色山峰 (病人)
surf2 = ax.plot_surface(X, Y, Z_sick, cmap='Reds', alpha=0.6, linewidth=0.2, edgecolors='red')

# 3. 绘制底部的等高线 (相当于把山峰压扁看地图)
ax.contour(X, Y, Z_healthy, zdir='z', offset=-0.05, cmap='Blues')
ax.contour(X, Y, Z_sick, zdir='z', offset=-0.05, cmap='Reds')

# 设置视角和标签
ax.view_init(elev=30, azim=-60)
ax.set_xlabel('Feature 1 (e.g., Body Temp)', fontsize=11, labelpad=10)
ax.set_ylabel('Feature 2 (e.g., WBC Count)', fontsize=11, labelpad=10)
ax.set_zlabel('Probability Density', fontsize=11, labelpad=10)
ax.set_title('3D Class Separation: Impact of Two Variables', fontsize=14)

# 手动添加图例 (3D图例比较麻烦，用Proxy)
import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color='blue', label='Healthy (Negatives)')
red_patch = mpatches.Patch(color='red', label='Sick (Positives)')
plt.legend(handles=[blue_patch, red_patch], loc='upper left')

plt.tight_layout()
plt.show()
