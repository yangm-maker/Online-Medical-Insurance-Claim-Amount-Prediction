# pages/0_数据分析图表展示.py
import streamlit as st
import os
from utils import set_chinese_font, load_all_resources

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
    .stImage {
        text-align: center !important;
    }
    </style>
""", unsafe_allow_html=True)

if resources is None:
    st.error(f"❌ {load_status}")
    st.stop()

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="数据分析图表",
    page_icon="📊",
    layout="wide"
)

# ---------------------- 主内容 ----------------------
st.title("📊 数据分析图表展示")
st.divider()

# 图表文件列表（按照数据分析脚本生成的顺序）
chart_files = [
    "1_总赔付金额分布直方图.png",
    "2_总赔付金额累积分布曲线.png",
    "3_年度医疗成本与总赔付金额散点图.png",
    "4_健康风险与总赔付金额箱线图.png",
    "5_吸烟状态与总赔付金额箱线图.png",
    "6_年龄分组与总赔付金额箱线图.png",
    "7_年龄与健康风险交叉分析.png",
    "8_保险计划与健康风险交叉分析.png"
]

# 图表说明
chart_descriptions = {
    "1_总赔付金额分布直方图.png": "展示总赔付金额的分布情况，包含正常值和异常值的区分，以及均值、中位数的标注",
    "2_总赔付金额累积分布曲线.png": "显示总赔付金额的累积分布，标注了25%、75%和95%分位数",
    "3_年度医疗成本与总赔付金额散点图.png": "分析年度医疗成本与总赔付金额的相关性，包含趋势线和相关系数",
    "4_健康风险与总赔付金额箱线图.png": "比较不同健康风险等级（基于慢性病数量）的总赔付金额分布",
    "5_吸烟状态与总赔付金额箱线图.png": "分析不同吸烟状态对总赔付金额的影响，标注中位数倍数关系",
    "6_年龄分组与总赔付金额箱线图.png": "展示不同年龄分组的赔付金额分布，显示年龄与赔付金额的相关性",
    "7_年龄与健康风险交叉分析.png": "双因素交叉分析，展示年龄和健康风险对赔付金额的协同影响",
    "8_保险计划与健康风险交叉分析.png": "双因素交叉分析，展示保险计划等级和健康风险对赔付金额的协同影响"
}

# 检查图表文件是否存在
available_charts = [f for f in chart_files if os.path.exists(f)]

if not available_charts:
    st.warning("⚠️ 未找到数据分析图表文件")
    st.markdown("""
    ### 如何生成图表？
    请先运行 `数据分析.py` 脚本生成分析图表：
    ```bash
    python 数据分析.py
    ```

    ### 预期生成的文件：
    - 1_总赔付金额分布直方图.png
    - 2_总赔付金额累积分布曲线.png
    - 3_年度医疗成本与总赔付金额散点图.png
    - 4_健康风险与总赔付金额箱线图.png
    - 5_吸烟状态与总赔付金额箱线图.png
    - 6_年龄分组与总赔付金额箱线图.png
    - 7_年龄与健康风险交叉分析.png
    - 8_保险计划与健康风险交叉分析.png
    """)
else:
    st.success(f"✅ 找到 {len(available_charts)} 个数据分析图表")

    # 图表选择下拉菜单
    selected_chart = st.selectbox(
        "选择要查看的分析图表",
        available_charts,
        format_func=lambda x: f"{x.split('_')[0]}. {x.split('_')[1].replace('.png', '')}"
    )

    # 显示选中的图表
    st.subheader(f"📈 {selected_chart.split('_')[1].replace('.png', '')}")

    # 图表说明
    if selected_chart in chart_descriptions:
        st.info(f"📋 **图表说明**：{chart_descriptions[selected_chart]}")

    # 显示图表
    try:
        st.image(selected_chart, use_container_width=True)

        # 分析洞察
        st.subheader("🔍 关键洞察")
        insights = {
            "1_总赔付金额分布直方图.png": """
            - **分布特征**：赔付金额呈现显著右偏分布，大部分用户赔付金额较低
            - **异常值**：存在少量高赔付用户，对整体均值产生较大影响
            - **业务意义**：需要重点关注高赔付用户的特征和风险控制
            """,
            "2_总赔付金额累积分布曲线.png": """
            - **集中程度**：80%的用户集中在相对较低的赔付区间
            - **长尾效应**：20%的高赔付用户贡献了较大比例的赔付总额
            - **风险分层**：可根据分位数进行用户风险等级划分
            """,
            "3_年度医疗成本与总赔付金额散点图.png": """
            - **强相关性**：年度医疗成本与赔付金额呈强正相关关系
            - **预测价值**：医疗成本是赔付金额的重要预测指标
            - **线性趋势**：两者之间存在明显的线性增长趋势
            """,
            "4_健康风险与总赔付金额箱线图.png": """
            - **风险梯度**：健康风险等级越高，赔付金额中位数和波动范围越大
            - **关键因素**：慢性病数量是赔付金额的核心驱动因素
            - **风控重点**：高风险人群需要更精细化的风险管理
            """,
            "5_吸烟状态与总赔付金额箱线图.png": """
            - **显著差异**：当前吸烟者的赔付金额明显高于从不吸烟者
            - **健康影响**：吸烟状态对健康风险和赔付成本有显著影响
            - **定价参考**：可作为保险费率差异化的重要依据
            """,
            "6_年龄分组与总赔付金额箱线图.png": """
            - **年龄效应**：随着年龄增长，赔付金额呈现上升趋势
            - **老年风险**：老年组的赔付金额中位数和波动性最高
            - **生命周期**：体现了医疗保险风险随年龄变化的特征
            """,
            "7_年龄与健康风险交叉分析.png": """
            - **协同效应**：年龄和健康风险存在明显的交互作用
            - **高风险组合**：老年+高风险用户的赔付金额显著高于其他组合
            - **精准风控**：需要针对特定年龄-风险组合制定风控策略
            """,
            "8_保险计划与健康风险交叉分析.png": """
            - **选择偏误**：高风险用户更倾向于选择高端保险计划
            - **逆向选择**：可能存在高风险用户选择更全面保障的现象
            - **产品设计**：需要优化保险产品设计以平衡风险和收益
            """
        }

        if selected_chart in insights:
            st.markdown(insights[selected_chart])

    except Exception as e:
        st.error(f"❌ 无法加载图表：{str(e)}")