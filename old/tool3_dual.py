
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# =========================
# 工具函数：3D 高斯“云团/山峰”
# =========================

def _regularize_cov(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    cov = cov + np.eye(3) * eps
    return cov

def _bhattacharyya_coefficient(mu0, cov0, mu1, cov1) -> float:
    """两个多元高斯的 Bhattacharyya 系数 BC ∈ (0,1]，越大越重合（更像）"""
    mu0 = np.asarray(mu0, dtype=float).reshape(3, 1)
    mu1 = np.asarray(mu1, dtype=float).reshape(3, 1)
    cov0 = _regularize_cov(cov0)
    cov1 = _regularize_cov(cov1)
    cov = (cov0 + cov1) / 2.0

    try:
        inv_cov = np.linalg.inv(cov)
        det_cov = float(np.linalg.det(cov))
        det0 = float(np.linalg.det(cov0))
        det1 = float(np.linalg.det(cov1))
    except np.linalg.LinAlgError:
        # 极端情况下退化：给一个很保守的值
        return 0.0

    d = (mu1 - mu0)
    term1 = 0.125 * float(d.T @ inv_cov @ d)  # (1/8) Δ^T Σ^{-1} Δ
    # (1/2) ln(det Σ / sqrt(det Σ0 det Σ1))
    term2 = 0.5 * np.log(max(det_cov, 1e-18) / max(np.sqrt(det0 * det1), 1e-18))
    DB = term1 + term2
    BC = float(np.exp(-DB))
    # 数值安全
    return float(np.clip(BC, 0.0, 1.0))

def _ellipsoid_surface(mu, cov, k=2.0, n_u=40, n_v=22):
    """
    生成椭球面（kσ 等密度壳）的 Surface 网格。
    x = mu + k * L * u, 其中 u 为单位球面点，L 为 cov 的 Cholesky
    """
    mu = np.asarray(mu, dtype=float).reshape(3, 1)
    cov = _regularize_cov(cov, eps=1e-6)

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # 退化时用特征分解兜底
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 1e-10, None)
        L = V @ np.diag(np.sqrt(w))

    u = np.linspace(0, 2*np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v)

    # 单位球
    xs = np.cos(uu) * np.sin(vv)
    ys = np.sin(uu) * np.sin(vv)
    zs = np.cos(vv)

    pts = np.stack([xs, ys, zs], axis=0).reshape(3, -1)  # 3 x (n_u*n_v)
    ell = mu + (k * L @ pts)

    X = ell[0, :].reshape(n_v, n_u)
    Y = ell[1, :].reshape(n_v, n_u)
    Z = ell[2, :].reshape(n_v, n_u)
    return X, Y, Z

def _sample_gaussian(mu, cov, n=350, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float).reshape(3,)
    cov = _regularize_cov(cov, eps=1e-6)
    return rng.multivariate_normal(mean=mu, cov=cov, size=n)

def _clamp_0_10(x):
    return float(np.clip(x, 0.0, 10.0))


# =========================
# 页面配置
# =========================
st.set_page_config(page_title="可视化沟通工具箱", layout="wide")

# 顶部选择：两个逻辑（你原来的地形 + 新的双山峰画像）
with st.sidebar:
    st.header("功能选择")
    tool = st.radio(
        "你想做哪类图？",
        ["体验画像对比（双山峰）", "数据地形探索（曲面）"],
        index=0
    )

# ==========================================================
# 1) 体验画像对比（双山峰）：简单版 + 复杂版 可切换
# ==========================================================
if tool == "体验画像对比（双山峰）":
    st.title("🎮 体验画像对比器（双山峰）")
    st.markdown("把 **当前体验** 和 **理想期待** 各画成一个“云团/山峰”，两者越重叠，代表越接近你想要的感觉。")

    # 轴名（都用 0-10）
    with st.sidebar:
        st.header("轴名（你可以随便改）")
        c1, c2, c3 = st.columns(3)
        with c1:
            name_x = st.text_input("X 轴", "剧情")
        with c2:
            name_y = st.text_input("Y 轴", "动作")
        with c3:
            name_z = st.text_input("Z 轴", "耐心")

        st.write("---")
        st.header("两个版本")
        mode = st.radio(
            "先体验哪个？",
            ["展示版（手填即可）", "深度版（可导入问卷/统计）"],
            index=0
        )

    # 通用：当前体验（场景①B：只填一组分数 + 容忍度）
    st.subheader("① 当前体验（你感受到的）")
    colA, colB = st.columns([2, 1], vertical_alignment="top")
    with colA:
        a1, a2, a3 = st.columns(3)
        with a1:
            cur_x = st.slider(f"{name_x}", 0.0, 10.0, 4.0, 0.1)
        with a2:
            cur_y = st.slider(f"{name_y}", 0.0, 10.0, 8.0, 0.1)
        with a3:
            cur_z = st.slider(f"{name_z}", 0.0, 10.0, 3.0, 0.1)

    with colB:
        sigma = st.slider("容忍度（云团胖瘦）", 0.4, 4.0, 1.4, 0.1)
        show_cloud = st.toggle("显示云团点（更像山峰）", value=True)
        shell_k = st.slider("显示范围（壳）", 1.5, 3.0, 2.0, 0.1)
        st.caption("容忍度越小=越挑剔（山峰更尖）；越大=越随和（云团更胖）。")

    mu_cur = np.array([cur_x, cur_y, cur_z], dtype=float)
    cov_cur = (sigma ** 2) * np.eye(3)

    # -------- 展示版：理想期待 = 手填一组（场景③A） --------
    if mode == "展示版（手填即可）":
        st.subheader("② 理想期待（你想要的）")
        b1, b2, b3 = st.columns(3)
        with b1:
            ideal_x = st.slider(f"{name_x}（理想）", 0.0, 10.0, 8.0, 0.1)
        with b2:
            ideal_y = st.slider(f"{name_y}（理想）", 0.0, 10.0, 8.0, 0.1)
        with b3:
            ideal_z = st.slider(f"{name_z}（理想）", 0.0, 10.0, 8.0, 0.1)

        mu_ideal = np.array([ideal_x, ideal_y, ideal_z], dtype=float)
        cov_ideal = cov_cur.copy()  # 展示版：两边用同一个“胖瘦”更容易解释

        source_note = "理想画像：手填"
        spread_note = None

    # -------- 深度版：理想期待 = 第二份 CSV（场景③B） --------
    else:
        st.subheader("② 理想期待（来自问卷/统计/第二份表）")

        with st.sidebar:
            st.header("深度版：导入理想画像")
            ideal_file = st.file_uploader("上传【理想期待】CSV（可选）", type=["csv"], key="ideal_csv")

        mu_ideal = None
        cov_ideal = None
        source_note = "理想画像：尚未导入（可先用手填兜底）"
        spread_note = None

        if ideal_file is not None:
            ideal_df = pd.read_csv(ideal_file)
            numeric_cols = ideal_df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 3:
                st.warning("这份表里可用的数字列不足 3 列：先换一份表，或先用手填兜底。")
            else:
                st.caption("请选择这份表里对应三个维度的列（0-10 分最好；如果不是也能用，但建议你先把量表统一）。")
                c1, c2, c3 = st.columns(3)
                with c1:
                    col_x = st.selectbox(f"{name_x} 对应列", numeric_cols, index=0, key="ideal_x")
                with c2:
                    col_y = st.selectbox(f"{name_y} 对应列", numeric_cols, index=1, key="ideal_y")
                with c3:
                    col_z = st.selectbox(f"{name_z} 对应列", numeric_cols, index=2, key="ideal_z")

                sub = ideal_df[[col_x, col_y, col_z]].dropna().copy()
                if len(sub) < 8:
                    st.warning("有效数据点太少（<8 行），建议更多样本，云团才稳定。")
                else:
                    # 计算均值与协方差（并把量表裁剪到 0-10 显示范围）
                    arr = sub.to_numpy(dtype=float)
                    arr = np.clip(arr, 0.0, 10.0)

                    mu_ideal = arr.mean(axis=0)
                    cov_ideal = np.cov(arr.T)

                    # 一个“人话”提示：大家分歧大不大（用标准差粗略表达）
                    stds = arr.std(axis=0)
                    spread = float(np.mean(stds))
                    if spread < 1.2:
                        spread_note = "理想期待很集中：大家想法比较一致。"
                    elif spread < 2.0:
                        spread_note = "理想期待有分歧：不同玩家想法不完全一样。"
                    else:
                        spread_note = "理想期待很分散：玩家想法差异较大。"

                    source_note = f"理想画像：来自 CSV（{len(arr)} 条）"

        # 没导入时，用手填兜底（仍然满足深度版可用）
        if mu_ideal is None or cov_ideal is None:
            st.caption("还没导入也没关系：你可以先用手填一组，演示效果不受影响。")
            b1, b2, b3 = st.columns(3)
            with b1:
                ideal_x = st.slider(f"{name_x}（理想-兜底）", 0.0, 10.0, 8.0, 0.1, key="ideal_fallback_x")
            with b2:
                ideal_y = st.slider(f"{name_y}（理想-兜底）", 0.0, 10.0, 8.0, 0.1, key="ideal_fallback_y")
            with b3:
                ideal_z = st.slider(f"{name_z}（理想-兜底）", 0.0, 10.0, 8.0, 0.1, key="ideal_fallback_z")

            mu_ideal = np.array([ideal_x, ideal_y, ideal_z], dtype=float)
            cov_ideal = cov_cur.copy()
            source_note = "理想画像：手填（兜底）"

    # ③ 计算“重合度”（界面只展示一个数 + 一句建议）
    overlap = _bhattacharyya_coefficient(mu_cur, cov_cur, mu_ideal, cov_ideal)
    overlap_pct = overlap * 100.0

    # 偏离最大的维度（给策划一句话就能懂）
    delta = mu_ideal - mu_cur
    names = [name_x, name_y, name_z]
    idx = int(np.argmax(np.abs(delta)))
    worst_dim = names[idx]
    worst_gap = float(delta[idx])

    # ============ 绘图（两个云团） ============
    fig = go.Figure()

    # 椭球壳
    X1, Y1, Z1 = _ellipsoid_surface(mu_cur, cov_cur, k=shell_k)
    fig.add_trace(go.Surface(
        x=X1, y=Y1, z=Z1,
        opacity=0.35,
        showscale=False,
        name="当前体验（云团）"
    ))

    X2, Y2, Z2 = _ellipsoid_surface(mu_ideal, cov_ideal, k=shell_k)
    fig.add_trace(go.Surface(
        x=X2, y=Y2, z=Z2,
        opacity=0.35,
        showscale=False,
        name="理想期待（云团）"
    ))

    # 云团点（更像山峰）
    if show_cloud:
        pts1 = _sample_gaussian(mu_cur, cov_cur, n=420, seed=1)
        pts2 = _sample_gaussian(mu_ideal, cov_ideal, n=420, seed=2)

        fig.add_trace(go.Scatter3d(
            x=pts1[:,0], y=pts1[:,1], z=pts1[:,2],
            mode="markers",
            marker=dict(size=2, opacity=0.15),
            name="当前（点）"
        ))
        fig.add_trace(go.Scatter3d(
            x=pts2[:,0], y=pts2[:,1], z=pts2[:,2],
            mode="markers",
            marker=dict(size=2, opacity=0.15),
            name="理想（点）"
        ))

    # 均值点
    fig.add_trace(go.Scatter3d(
        x=[mu_cur[0]], y=[mu_cur[1]], z=[mu_cur[2]],
        mode="markers+text",
        marker=dict(size=6),
        text=["当前"],
        textposition="top center",
        name="当前中心"
    ))
    fig.add_trace(go.Scatter3d(
        x=[mu_ideal[0]], y=[mu_ideal[1]], z=[mu_ideal[2]],
        mode="markers+text",
        marker=dict(size=6),
        text=["理想"],
        textposition="top center",
        name="理想中心"
    ))

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis=dict(title=name_x, range=[0, 10]),
            yaxis=dict(title=name_y, range=[0, 10]),
            zaxis=dict(title=name_z, range=[0, 10]),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============ 给“人话结果” ============
    st.markdown("### ✅ 一眼结论")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("重合度", f"{overlap_pct:.1f}%")
    with c2:
        st.metric("差距最大维度", worst_dim)
    with c3:
        st.metric("差距方向", "理想更高" if worst_gap > 0 else ("当前更高" if worst_gap < 0 else "几乎一样"))

    # 简短建议（不讲统计）
    if overlap_pct >= 75:
        st.success("整体很接近：你更多是在“微调口味”。可以聚焦差距最大的那一轴，提几个具体例子给策划。")
    elif overlap_pct >= 45:
        st.warning("有明显差距：建议优先围绕差距最大维度，说清楚“现在在哪、我想要哪”。")
    else:
        st.error("差距较大：更像是两个不同方向的体验。建议你先明确哪一轴是你最核心的诉求，再谈其它。")

    st.caption(f"{source_note}" + (f"｜{spread_note}" if spread_note else ""))

    with st.expander("（可选）把这段话复制给策划"):
        # 给用户一段可复制“说明文字”，方便沟通
        cur_txt = f"当前体验中心：{name_x}={mu_cur[0]:.1f}, {name_y}={mu_cur[1]:.1f}, {name_z}={mu_cur[2]:.1f}"
        ideal_txt = f"理想期待中心：{name_x}={mu_ideal[0]:.1f}, {name_y}={mu_ideal[1]:.1f}, {name_z}={mu_ideal[2]:.1f}"
        overlap_txt = f"两者重合度约 {overlap_pct:.1f}%。最大差距在「{worst_dim}」：理想比当前 {'高' if worst_gap>0 else '低'} {abs(worst_gap):.1f} 分。"
        st.text("\n".join([cur_txt, ideal_txt, overlap_txt]))

# ==========================================================
# 2) 数据地形探索（曲面）：保留你原来的逻辑（可做备选）
# ==========================================================
else:
    st.title("🏔️ 数据地形探索器（曲面）")
    st.markdown("用 AI 拟合一个平滑的 3D 地形图，观察两个因素如何共同影响一个结果。")

    with st.sidebar:
        st.header("1. 上传数据")
        uploaded_file = st.file_uploader("上传你的 CSV 表格", type=["csv"], key="terrain_csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 3:
            st.error("数据太少啦！至少需要 3 列数值数据才能构建 3D 模型。")
            st.stop()

        with st.sidebar:
            st.header("2. 定义坐标轴")
            col1, col2, col3 = st.columns(3)
            with col1:
                x_axis = st.selectbox("X 轴", numeric_cols, index=0, key="x_axis")
            with col2:
                y_axis = st.selectbox("Y 轴", numeric_cols, index=1, key="y_axis")
            with col3:
                z_axis = st.selectbox("结果 Z", numeric_cols, index=2, key="z_axis")

            st.write("---")
            n_estimators = st.slider("地形稳定度", 50, 300, 120, step=10, key="n_estimators")
            grid_n = st.slider("地形精细度", 20, 80, 35, step=5, key="grid_n")

        df_clean = df.dropna(subset=[x_axis, y_axis, z_axis]).copy()
        X_train = df_clean[[x_axis, y_axis]]
        y_train = df_clean[z_axis]

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        x_range = np.linspace(df_clean[x_axis].min(), df_clean[x_axis].max(), grid_n)
        y_range = np.linspace(df_clean[y_axis].min(), df_clean[y_axis].max(), grid_n)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        zz = model.predict(grid_points).reshape(xx.shape)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=df_clean[x_axis], y=df_clean[y_axis], z=df_clean[z_axis],
            mode='markers',
            marker=dict(size=4, color='black', opacity=0.5),
            name='原始数据点'
        ))
        fig.add_trace(go.Surface(
            z=zz, x=x_range, y=y_range,
            colorscale='Viridis',
            opacity=0.8,
            name='预测曲面'
        ))

        fig.update_layout(
            title=f"3D 视图：{x_axis} + {y_axis} 影响 {z_axis}",
            scene=dict(
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                zaxis_title=z_axis
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=650
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"💡 你可以旋转上方图表查看 {x_axis} 和 {y_axis} 的不同组合如何改变 {z_axis}。")

    else:
        st.info("👈 请在左侧上传 CSV 文件开始体验。")
        st.markdown("### 示例数据格式：")
        st.table(pd.DataFrame({
            '游戏难度': [1, 2, 3, 8, 9],
            '投入成本': [10, 20, 30, 80, 90],
            '玩家人数': [100, 200, 150, 50, 20]
        }))
