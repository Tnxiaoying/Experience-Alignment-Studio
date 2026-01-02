import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# --- 1. 页面配置 (美观基础) ---
st.set_page_config(page_title="数据地形探索器", layout="wide")

st.title("🏔️ Data Landscape Explorer")
st.markdown("通过 AI 拟合，探索任意三个变量之间的 **响应面 (Response Surface)** 关系。")

# --- 2. 侧边栏：数据上传与设置 ---
with st.sidebar:
    st.header("1. 上传数据")
    uploaded_file = st.file_uploader("上传你的 CSV 表格", type=["csv"])
    
    # 初始化一些变量
    df = None
    model = None

# --- 3. 核心逻辑 ---
if uploaded_file is not None:
    # 读取数据
    df = pd.read_csv(uploaded_file)
    
    # 过滤出数值型列 (只有数字才能画坐标轴)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 3:
        st.error("数据太少啦！至少需要3列数值数据才能构建3D模型。")
    else:
        # --- 侧边栏：选择轴 ---
        with st.sidebar:
            st.header("2. 定义坐标轴")
            # 让用户选择 X, Y, Z，默认选前三列
            col1, col2, col3 = st.columns(3)
            x_axis = st.selectbox("选择 X 轴 (自变量1)", numeric_cols, index=0)
            y_axis = st.selectbox("选择 Y 轴 (自变量2)", numeric_cols, index=1)
            z_axis = st.selectbox("选择 Z 轴 (目标/结果)", numeric_cols, index=2)
            
            st.write("---")
            st.caption("提示：模型会自动学习 X 和 Y 如何共同影响 Z。")

        # --- 4. 机器学习：训练模型 (造山峰) ---
        # 准备数据
        X_train = df[[x_axis, y_axis]]
        y_train = df[z_axis]
        
        # 实例化算法：使用随机森林回归 (鲁棒性强，甚至能拟合非线性关系)
        # 这里就是体现你“思维能力”的地方：不仅仅是画散点，而是用算法寻找规律
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # --- 5. 生成网格数据 (为了画平滑的曲面) ---
        # 在 X 和 Y 的范围内生成 30x30 的网格
        x_range = np.linspace(df[x_axis].min(), df[x_axis].max(), 30)
        y_range = np.linspace(df[y_axis].min(), df[y_axis].max(), 30)
        xx, yy = np.meshgrid(x_range, y_range)
        
        # 让模型预测网格上每个点的 Z 值
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predicted_z = model.predict(grid_points)
        zz = predicted_z.reshape(xx.shape)

        # --- 6. 可视化：绘制交互式 3D 图 ---
        # 创建 Plotly 图表
        fig = go.Figure()

        # 层1：绘制真实的原始数据散点 (让用户看清原始分布)
        fig.add_trace(go.Scatter3d(
            x=df[x_axis], y=df[y_axis], z=df[z_axis],
            mode='markers',
            marker=dict(size=4, color='black', opacity=0.5),
            name='原始数据点'
        ))

        # 层2：绘制 AI 拟合的曲面 (Landscape)
        fig.add_trace(go.Surface(
            z=zz, x=x_range, y=y_range,
            colorscale='Viridis',
            opacity=0.8,
            name='预测曲面'
        ))

        # 美化图表布局
        fig.update_layout(
            title=f"3D 视图: {x_axis} & {y_axis} -> {z_axis}",
            scene=dict(
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                zaxis_title=z_axis
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=600
        )

        # 在页面展示
        st.plotly_chart(fig, use_container_width=True)
        
        # --- 7. 显示统计信息 (增加专业度) ---
        st.info(f"💡 模型解释：根据当前数据，我们构建了一个地形图。你可以旋转上方图表查看 {x_axis} 和 {y_axis} 的不同组合如何改变 {z_axis}。")

else:
    # 引导页面 (当没传文件时显示)
    st.info("👈 请在左侧上传 CSV 文件开始体验。")
    # 造一些假数据做演示，避免空白太丑
    st.markdown("### 示例数据格式：")
    example_df = pd.DataFrame({
        '游戏难度': [1, 2, 3, 8, 9],
        '投入成本': [10, 20, 30, 80, 90],
        '玩家人数': [100, 200, 150, 50, 20]
    })
    st.table(example_df)
