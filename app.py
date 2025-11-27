# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import set_chinese_font, load_all_resources

# ---------------------- 初始化 ----------------------
set_chinese_font()
resources, load_status = load_all_resources()

# 注入CSS控制图片显示大小
st.markdown("""
    <style>
    /* 限制所有图片的最大尺寸 */
    img {
        max-width: 800px !important;
        max-height: 600px !important;
        height: auto !important;
        margin: 0 auto !important;
    }
    .stPlotlyChart, .stPyplot {
        max-width: 800px !important;
        margin: 0 auto !important;
    }
    </style>
""", unsafe_allow_html=True)

# 加载失败处理
if resources is None:
    st.error(f"❌ {load_status}")
    st.stop()

# 提取资源
raw_data = resources["raw_data"]
detailed_results = resources["detailed_results"]
target_col = "total_claims_paid"  # 目标变量名

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="医疗保险索赔预测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- 侧边栏 ----------------------
st.sidebar.title("🏥 医疗保险索赔预测")
st.sidebar.divider()
st.sidebar.info("📋 系统说明：基于机器学习预测医疗保险索赔金额，支持数据可视化与在线预测。")

# ---------------------- 主内容：数据概览 ----------------------
st.title("📊 数据概览")
st.divider()

# 1. 数据基本信息
if raw_data is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("数据总量", f"{raw_data.shape[0]:,} 条")
    with col2:
        st.metric("特征数量", f"{raw_data.shape[1]} 个")
    with col3:
        st.metric("目标变量", "索赔金额")

    # 2. 原始数据预览
    st.subheader("📋 原始数据预览（前10条）")
    st.dataframe(raw_data.head(10), use_container_width=True)

    # 3. 图表选择下拉菜单
    st.subheader("📈 数据可视化")

    chart_option = st.selectbox(
        "选择要查看的图表",
        [
            "🎯 索赔金额分布",
            "🔗 特征相关性热力图"
        ],
        index=0
    )

    if chart_option == "🎯 索赔金额分布":
        if target_col in raw_data.columns:
            # 减小图形尺寸
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            # 剔除极端值后的直方图
            clip_max = raw_data[target_col].quantile(0.95)
            ax1.hist(
                raw_data[target_col].clip(0, clip_max),
                bins=40, color='skyblue', alpha=0.7, edgecolor='black'
            )
            ax1.set_title(f"索赔金额分布（剔除前5%极端值）", fontsize=12)
            ax1.set_xlabel("索赔金额（元）", fontsize=10)
            ax1.set_ylabel("频数", fontsize=10)
            ax1.tick_params(axis='both', which='major', labelsize=9)
            ax1.grid(alpha=0.3)

            # 统计信息文本
            zero_ratio = (raw_data[target_col] == 0).mean()
            stats_text = f"""
            统计摘要：
            • 零索赔比例：{zero_ratio:.2%}
            • 平均索赔金额：{raw_data[target_col].mean():.2f} 元
            • 中位数索赔金额：{raw_data[target_col].median():.2f} 元
            • 最大索赔金额：{raw_data[target_col].max():.2f} 元
            • 最小索赔金额：{raw_data[target_col].min():.2f} 元
            """
            ax2.axis('off')
            ax2.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center')

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    elif chart_option == "🔗 特征相关性热力图":
        numeric_cols = [
            "age", "income", "bmi", "risk_score", "chronic_count",
            "deductible", "annual_premium", target_col
        ]
        valid_cols = [col for col in numeric_cols if col in raw_data.columns]
        if len(valid_cols) > 2:
            # 减小图形尺寸
            fig, ax = plt.subplots(figsize=(7, 5))
            corr_matrix = raw_data[valid_cols].corr()
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, shrink=0.8)

            # 坐标轴标签
            ax.set_xticks(range(len(valid_cols)))
            ax.set_yticks(range(len(valid_cols)))
            ax.set_xticklabels(valid_cols, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(valid_cols, fontsize=9)

            # 添加相关系数文本
            for i in range(len(valid_cols)):
                for j in range(len(valid_cols)):
                    ax.text(
                        j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=8
                    )

            ax.set_title("关键特征相关性热力图", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.warning("⚠️ 没有足够的数据列来生成相关性热力图")

else:
    st.warning("⚠️ 未找到原始数据文件（medical_insurance.csv），部分功能受限")