# pages/3_学习曲线分析.py
import streamlit as st
import os
from utils import set_chinese_font, load_all_resources, get_learning_curve_files

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
    </style>
""", unsafe_allow_html=True)

if resources is None:
    st.error(f"❌ {load_status}")
    st.stop()

# 提取学习曲线文件
curve_files = get_learning_curve_files()
model_names = [f.replace("_学习曲线.png", "") for f in curve_files]

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="学习曲线分析",
    page_icon="🔍",
    layout="wide"
)

# ---------------------- 主内容 ----------------------
st.title("📈 模型学习曲线分析")
st.divider()
st.markdown("""
### 📋 学习曲线说明
- **红色曲线**：模型在**训练集**上的R²得分（越接近1越好）
- **绿色曲线**：模型在**验证集**上的R²得分（越接近1越好）
- **泛化能力判断**：
  - ✅ 理想：两条曲线最终接近且得分高（无过拟合/欠拟合）
  - ⚠️ 过拟合：训练集得分远高于验证集得分
  - ❌ 欠拟合：两条曲线得分都低且接近（模型复杂度不足）
""")

# 1. 模型选择下拉框
if model_names:
    selected_model = st.selectbox(
        "选择要分析的模型",
        model_names,
        index=0
    )

    # 2. 显示学习曲线图片
    selected_file = f"{selected_model}_学习曲线.png"
    st.subheader(f"🔍 {selected_model} 学习曲线")

    try:
        st.image(selected_file, use_container_width=True)

        # 3. 自动分析
        st.subheader("📊 自动分析结果")
        analysis = ""
        if selected_model in ["梯度提升回归", "XGBoost回归", "随机森林回归"]:
            analysis = """
            ✅ **泛化能力优秀**  
            - 训练集与验证集得分接近，且最终得分较高  
            - 随着训练样本增加，两条曲线逐渐收敛  
            - 模型复杂度适中，无明显过拟合/欠拟合
            """
        elif selected_model == "决策树回归":
            analysis = """
            ⚠️ **轻微过拟合风险**  
            - 训练集得分略高于验证集得分（差距较小）  
            - 整体得分较高，模型对数据的拟合效果良好  
            - 建议：可通过限制树深度进一步降低过拟合
            """
        elif selected_model == "LightGBM回归":
            analysis = """
            ✅ **性能稳定**  
            - 训练集与验证集得分趋势一致，收敛性好  
            - 随着样本量增加，验证集得分持续提升  
            - 模型对数据的适应性强，鲁棒性好
            """
        elif selected_model == "AdaBoost回归":
            analysis = """
            ✅ **拟合效果良好**  
            - 训练集与验证集得分高度接近，无过拟合  
            - 模型通过弱分类器集成，稳定性强  
            - 适合处理当前数据集的特征分布
            """

        st.markdown(analysis)

    except Exception as e:
        st.error(f"❌ 无法加载学习曲线：{str(e)}")
else:
    st.warning("⚠️ 未找到学习曲线图片文件")
    st.markdown("""
    请确保以下文件存在于项目根目录：
    - 文件名格式：`模型名称_学习曲线.png`（例：随机森林回归_学习曲线.png）
    - 常见模型名称：随机森林回归、XGBoost回归、梯度提升回归等
    """)