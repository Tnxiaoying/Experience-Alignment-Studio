import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor


# --- 小工具：缓存读入与训练，避免每次拖动控件都重训 ---
@st.cache_data(show_spinner=False)
def _load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_resource(show_spinner=False)
def _train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int, random_state: int):
    # oob_prediction_ 能提供一种“更接近实战”的预测（每个点只用没见过它的树来估计），
    # 用来做“误判/漏判”的体感提示更靠谱一些。
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        oob_score=True,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model

# --- 1. 页面配置 (美观基础) ---
st.set_page_config(page_title="数据地形探索器", layout="wide")

st.title("🏔️ 数据地形探索器")
st.markdown("用一个可旋转的 3D 地形图，快速感受 **两个因素如何共同影响一个结果**。")

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
    df = _load_csv(uploaded_file)
    
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
            with col1:
                x_axis = st.selectbox("X 轴", numeric_cols, index=0)
            with col2:
                y_axis = st.selectbox("Y 轴", numeric_cols, index=1)
            with col3:
                z_axis = st.selectbox("结果 Z", numeric_cols, index=2)

            st.write("---")
            st.header("3. 一个简单的“达标线”")
            st.caption("把结果 Z ≥ T 看作“达标/可做”。拖动 T 看看哪些点容易被误判或漏掉。")

            # 训练参数与显示选项（尽量少，避免界面堆砌）
            n_estimators = st.slider("地形稳定度", 50, 300, 120, step=10)
            grid_n = st.slider("地形精细度", 20, 80, 35, step=5)
            show_mistakes = st.toggle("标出易误判点", value=True)
            
            st.write("---")
            st.caption("提示：模型会自动学习 X 和 Y 如何共同影响 Z。")

        # --- 4. 机器学习：训练模型 (造山峰) ---
        # 清理缺失值（避免训练报错）
        df_clean = df.dropna(subset=[x_axis, y_axis, z_axis]).copy()
        if len(df_clean) < 10:
            st.error("可用数据点太少（去掉缺失值后不足 10 行），请换一份数据或补齐缺失值。")
            st.stop()

        # 准备数据
        X_train = df_clean[[x_axis, y_axis]]
        y_train = df_clean[z_axis]
        
        # 实例化算法：使用随机森林回归 (鲁棒性强，甚至能拟合非线性关系)
        # 这里就是体现你“思维能力”的地方：不仅仅是画散点，而是用算法寻找规律
        model = _train_model(X_train, y_train, n_estimators=n_estimators, random_state=42)
        
        # --- 5. 生成网格数据 (为了画平滑的曲面) ---
        # 在 X 和 Y 的范围内生成网格
        x_range = np.linspace(df_clean[x_axis].min(), df_clean[x_axis].max(), grid_n)
        y_range = np.linspace(df_clean[y_axis].min(), df_clean[y_axis].max(), grid_n)
        xx, yy = np.meshgrid(x_range, y_range)
        
        # 让模型预测网格上每个点的 Z 值
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predicted_z = model.predict(grid_points)
        zz = predicted_z.reshape(xx.shape)

        # --- 5.5 计算“达标线”相关的误判/漏判（只用通俗措辞呈现） ---
        # 这里用“模型在训练集上的预测”做一个快速体感指标：告诉你阈值放在这里时，哪里更容易看走眼。
        # （强调：这是辅助探索，不是最终结论。）
        default_T = float(np.nanmedian(y_train))
        with st.sidebar:
            T = st.slider(
                "达标阈值 T",
                float(y_train.min()),
                float(y_train.max()),
                default_T,
            )

        # 优先用 OOB（更接近“没见过该点”的预测）；如果因为数据太少出现 NaN，再退回普通预测
        y_hat = getattr(model, "oob_prediction_", None)
        if y_hat is None or np.any(np.isnan(y_hat)):
            y_hat = model.predict(X_train)
        true_ok = (y_train.values >= T)
        pred_ok = (y_hat >= T)
        miss = true_ok & (~pred_ok)   # 本来达标，但模型没看出来（漏掉）
        false_alarm = (~true_ok) & pred_ok  # 本来不达标，但模型以为达标（误判）

        miss_n = int(miss.sum())
        false_alarm_n = int(false_alarm.sum())
        total_n = int(len(y_train))

        # --- 6. 可视化：绘制交互式 3D 图 ---
        # 创建 Plotly 图表
        fig = go.Figure()

        # 层1：绘制真实的原始数据散点 (让用户看清原始分布)
        fig.add_trace(go.Scatter3d(
            x=df_clean[x_axis], y=df_clean[y_axis], z=df_clean[z_axis],
            mode='markers',
            marker=dict(size=4, color='black', opacity=0.5),
            name='原始数据点'
        ))

        # 标出“容易看走眼”的点（可关闭）
        if show_mistakes:
            # 误判为达标（误报）
            if false_alarm_n > 0:
                fig.add_trace(go.Scatter3d(
                    x=X_train.loc[false_alarm, x_axis],
                    y=X_train.loc[false_alarm, y_axis],
                    z=y_train.loc[false_alarm],
                    mode='markers',
                    marker=dict(size=7, color='#d62728', opacity=0.9),
                    name='误判为达标'
                ))
            # 漏掉达标点（漏报）
            if miss_n > 0:
                fig.add_trace(go.Scatter3d(
                    x=X_train.loc[miss, x_axis],
                    y=X_train.loc[miss, y_axis],
                    z=y_train.loc[miss],
                    mode='markers',
                    marker=dict(size=7, color='#1f77b4', opacity=0.9),
                    name='漏掉的达标点'
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
            title=f"3D 视图：{x_axis} + {y_axis} 影响 {z_axis}",
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

        # --- 7. 用“人话”给结论与建议（不堆统计术语） ---
        st.markdown("### ✅ 快速提示")
        st.write(
            f"当前把 **{z_axis} ≥ {T:.3g}** 视为‘达标/可做’。在现有数据里："
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("误判为达标", f"{false_alarm_n} 个")
        with c2:
            st.metric("漏掉的达标点", f"{miss_n} 个")
        with c3:
            st.metric("参与判断的数据点", f"{total_n} 个")

        # 简短建议：只给“下一步动作”，不讲术语
        if false_alarm_n == 0 and miss_n == 0:
            st.success("这个阈值下，模型对现有数据的判断很一致：暂时没看到明显‘误判/漏判’。你可以再换几个 T 看敏感性。")
        elif miss_n > false_alarm_n:
            st.warning("更容易**漏掉**本来达标的点：如果你更怕错过机会，可以把 T 稍微调低一点；或者在蓝点附近补一些数据，让地形更清晰。")
        elif false_alarm_n > miss_n:
            st.warning("更容易**误判**为达标：如果你更怕误投/误做，可以把 T 稍微调高一点；或者在红点附近补一些数据，减少看走眼。")
        else:
            st.info("误判和漏判差不多：你可以根据业务偏好选择——‘宁可多试错’就降低 T，‘宁可更保守’就提高 T。")

        with st.expander("想看一下具体是哪些点？（可选）"):
            tmp = df_clean[[x_axis, y_axis, z_axis]].copy()
            tmp["模型预测"] = y_hat
            tmp["标签"] = "正常"
            tmp.loc[false_alarm, "标签"] = "误判为达标"
            tmp.loc[miss, "标签"] = "漏掉的达标点"
            show_df = tmp[tmp["标签"] != "正常"].sort_values("标签")
            if len(show_df) == 0:
                st.write("当前没有需要特别关注的点。")
            else:
                st.dataframe(show_df, use_container_width=True)

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
