# pages/1_模型性能对比.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import set_chinese_font, load_all_resources, _get_model_type

# ---------------------- 初始化 ----------------------
set_chinese_font()
resources, load_status = load_all_resources()

# 注入CSS控制图片显示大小
st.markdown("""
    <style>
    img {
        max-width: 800px !important;
        max-height: 600px !important;
        height: auto !important;
        margin: 0 auto !important;
    }
    .stPyplot {
        max-width: 800px !important;
        margin: 0 auto !important;
    }
    </style>
""", unsafe_allow_html=True)

if resources is None:
    st.error(f"❌ {load_status}")
    st.stop()

# 提取资源
model_scores = resources["model_scores"]
detailed_results = resources["detailed_results"]
best_model_name = detailed_results.iloc[0]["模型"]  # 最优模型名称

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="模型性能对比",
    page_icon="📈",
    layout="wide"
)

# ---------------------- 主内容 ----------------------
st.title("📊 模型性能对比")
st.divider()

# 1. 性能指标说明
st.markdown("""
### 📋 性能指标说明
- **R² 得分**：越接近1越好，表示模型对数据的解释力
- **RMSE（均方根误差）**：越小越好，表示预测值与真实值的平均偏差
- **MAE（平均绝对误差）**：越小越好，表示预测值的平均绝对偏差
""")

# 2. 性能表格（按R²排序）
st.subheader("📋 模型性能详细排名")
st.dataframe(
    detailed_results[["模型", "R²", "RMSE", "MAE", "类型"]].round(4),
    use_container_width=True,
    column_config={
        "R²": st.column_config.NumberColumn(format="%.4f"),
        "RMSE": st.column_config.NumberColumn(format="%.2f"),
        "MAE": st.column_config.NumberColumn(format="%.2f")
    }
)

# 3. 图表选择下拉菜单
st.subheader("📈 性能可视化")

chart_option = st.selectbox(
    "选择要查看的性能图表",
    [
        "📊 R²得分排名",
        "📉 RMSE排名",
        "🎨 模型类型分布",
        "⭐ 最优模型性能"
    ],
    index=0
)

if chart_option == "📊 R²得分排名":
    st.subheader("📊 模型R²得分排名（前10）")

    top10_models = detailed_results.head(10)
    colors = [
        "purple" if t == "集成方法" else
        "orange" if t == "线性模型" else
        "brown" if t == "其他模型" else "skyblue"
        for t in top10_models["类型"]
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(top10_models)), top10_models["R²"], color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(range(len(top10_models)))
    ax.set_yticklabels(top10_models["模型"], fontsize=9)
    ax.set_xlabel("R² 得分", fontsize=10)
    ax.set_title("模型R²得分排名（前10）", fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='x', labelsize=9)

    # 添加数值标签
    for i, v in enumerate(top10_models["R²"]):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

elif chart_option == "📉 RMSE排名":
    st.subheader("📉 模型RMSE排名（前10）")

    top10_models = detailed_results.head(10)
    colors = [
        "purple" if t == "集成方法" else
        "orange" if t == "线性模型" else
        "brown" if t == "其他模型" else "skyblue"
        for t in top10_models["类型"]
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(top10_models)), top10_models["RMSE"], color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(range(len(top10_models)))
    ax.set_yticklabels(top10_models["模型"], fontsize=9)
    ax.set_xlabel("RMSE（元）", fontsize=10)
    ax.set_title("模型RMSE排名（前10）", fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='x', labelsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

elif chart_option == "🎨 模型类型分布":
    st.subheader("🎨 模型类型分布")

    type_counts = detailed_results["类型"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct='%1.1f%%',
        colors=["skyblue", "purple", "orange", "brown"],
        startangle=90,
        textprops={'fontsize': 10}
    )
    # 设置百分比文字大小
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax.set_title("模型类型分布", fontsize=12)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

elif chart_option == "⭐ 最优模型性能":
    st.subheader("⭐ 最优模型性能分析")

    best_scores = detailed_results.iloc[0]
    metrics = ["R²", "RMSE", "MAE"]
    raw_values = [best_scores["R²"], best_scores["RMSE"], best_scores["MAE"]]

    # 标准化RMSE/MAE（转为"越高越好"）
    max_rmse = detailed_results["RMSE"].max()
    max_mae = detailed_results["MAE"].max()
    norm_values = [
        raw_values[0],  # R²保持不变
        1 - (raw_values[1] / max_rmse),  # RMSE反向标准化
        1 - (raw_values[2] / max_mae)  # MAE反向标准化
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(metrics, norm_values, color=["green", "red", "orange"], alpha=0.8, width=0.6)
    ax.set_ylabel("标准化得分（越高越好）", fontsize=10)
    ax.set_title(f"最优模型：{best_scores['模型']} 性能", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='both', labelsize=9)

    # 添加原始数值标签
    for i, (n_val, r_val) in enumerate(zip(norm_values, raw_values)):
        ax.text(i, n_val + 0.02, f'原始值：{r_val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# 4. 最优模型详情
st.subheader("🔧 最优模型详细信息")
if "集成" not in best_model_name:
    # 非集成模型：显示参数
    best_params = model_scores[best_model_name]["最优参数"]
    st.markdown(f"**模型名称**：{best_model_name}")
    st.markdown(f"**模型类型**：{detailed_results.iloc[0]['类型']}")
    st.markdown("**最优训练参数**：")
    params_df = pd.DataFrame(list(best_params.items()), columns=["参数名", "参数值"])
    st.dataframe(params_df, use_container_width=True)
else:
    # 集成模型：显示说明
    st.markdown(f"**集成方法**：{best_model_name}")
    st.markdown("**集成逻辑**：基于多个基础模型的预测结果组合，降低单模型误差，提升稳定性")
    st.markdown(f"**基础模型数量**：{len([k for k in model_scores.keys() if '集成' not in k])} 个")