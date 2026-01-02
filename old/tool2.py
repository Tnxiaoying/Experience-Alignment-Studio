import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. 构建网格
# X轴: 难度 (0=无脑, 10=受苦)
difficulty = np.linspace(0, 10, 100)
# Y轴: 资源投入 (0=用爱发电, 10=数亿美金)
investment = np.linspace(0, 10, 100)
X, Y = np.meshgrid(difficulty, investment)

# 2. 模拟市场反馈函数 (Z轴: 潜在玩家数量/百万)
# 假设存在两个主要的成功模式 (两个山峰)

# 山峰1: 低投入、低难度的休闲爆款 (Casual Hit)
# 中心在 (2, 2)，范围较窄
z1 = 3 * np.exp(-((X - 2)**2 + (Y - 2)**2) / 3)

# 山峰2: 高投入、高难度的硬核大作 (Hardcore AAA)
# 中心在 (8, 8)，范围较宽
z2 = 5 * np.exp(-((X - 8)**2 + (Y - 8)**2) / 8)

# 这是一个复杂的市场，混合了这两个因素，还有一些底噪
Z = z1 + z2

# 3. 绘图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制地形表面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# 4. 模拟你的“尝试造一个山峰” (Project Simulation)
# 假设你想做一个“中等难度(5)，中等投入(5)”的游戏
my_game_difficulty = 5
my_game_investment = 5
# 计算预期玩家数
expected_players = 3 * np.exp(-((5 - 2)**2 + (5 - 2)**2) / 3) + 5 * np.exp(-((5 - 8)**2 + (5 - 8)**2) / 8)

ax.scatter(my_game_difficulty, my_game_investment, expected_players, color='red', s=200, label='Your Concept Project')
ax.text(my_game_difficulty, my_game_investment, expected_players + 0.5, 'Your Project\n(In the "Valley of Death"?)', color='black', fontsize=10)

# 标注两个成功的高地
ax.text(2, 2, 3.5, 'Casual Hit\n(Low Cost, Easy)', color='white', fontweight='bold', ha='center')
ax.text(8, 8, 5.5, 'Hardcore AAA\n(High Cost, Hard)', color='white', fontweight='bold', ha='center')

# 轴标签
ax.set_xlabel('Game Difficulty (X)', fontsize=11, labelpad=10)
ax.set_ylabel('Resource Investment (Y)', fontsize=11, labelpad=10)
ax.set_zlabel('Potential Players (Z)', fontsize=11, labelpad=10)
ax.set_title('Market Response Surface: Finding the Sweet Spot', fontsize=14)
ax.view_init(elev=30, azim=225) # 调整视角以便看清

plt.show()
