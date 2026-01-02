# 🎮 Experience Alignment Studio | 可视化沟通工具箱

> “拒绝形容词，用数据量化手感。”
> Stop arguing with adjectives. Align experiences with data.

## 🎯 核心愿景 (Vision)

在游戏开发过程中，策划与制作人、或开发团队与玩家之间，最大的成本往往来自于“体验认知的错位”：
* 策划说：“这游戏太难了。”
* 制作人说：“我觉得刚好，是你要的太简单。”

这种基于主观感觉的争论往往没有结果。Experience Alignment Studio (EAS) 是一个基于数据科学的可视化工具，旨在将模糊的“体验”转化为可度量的高维分布（High-Dimensional Distribution）。

我们不追求绝对的“正确答案”，而是追求“可视化的共识”。

---

## 🛠️ 功能模块 (Features)

### 1. 体验画像对比 (Experience Matching) —— *解决“感觉不对”*
利用 Bhattacharyya Coefficient (巴塔恰里亚系数) 算法，计算“当前版本”与“理想目标”之间的数学重合度。

* 双山峰可视化：将体验抽象为 3D 高斯云团。云团越胖，代表容忍度越高；云团越尖，代表要求越明确。
* 重合度量化：直接给出一个 0-100% 的对齐分数。
* 2D 切片分析：在 3D 看不清时，自动生成降维切片，精准定位分歧最大的维度（是数值填错了，还是机制设计偏了？）。

### 2. 决策地形探索 (Decision Landscape) —— *解决“盲目调整”*
利用 随机森林 (Random Forest) 机器学习算法，将离散的测试数据（如 Steam 销量、问卷评分）拟合为连续的 响应面 (Response Surface)。

* 寻找甜蜜点 (Sweet Spot)：可视化展示哪种“难度”与“资源投入”的组合能带来最高的玩家留存。
* 风险规避：识别数据中的“死亡谷”区域，避免无效的数值内卷。

---

## 🚀 快速开始 (Quick Start)

### 方式一：在线体验（推荐）
无需安装任何环境，直接访问网页版：
👉 [点击这里打开工作台](https://你的streamlit链接填在这里)

### 方式二：本地运行
如果你是开发者或需要处理敏感数据，可以在本地运行：

1. 安装依赖
   ```bash
   pip install -r requirements.txt
