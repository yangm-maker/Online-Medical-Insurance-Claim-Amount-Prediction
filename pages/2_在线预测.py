# pages/2_在线预测.py
import streamlit as st
import pandas as pd
import numpy as np
# 导入自定义工具函数（含新增的MySQL写入函数）
from utils import set_chinese_font, load_all_resources, quick_feature_engineering, save_prediction_to_mysql

# ---------------------- 初始化 ----------------------
set_chinese_font()
resources, load_status = load_all_resources()

# 注入CSS（保持一致性）
st.markdown("""
    <style>
    img {
        max-width: 800px !important;
        max-height: 600px !important;
        height: auto !important;
    }
    .uploadedFile {
        border: 2px dashed #ccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

if resources is None:
    st.error(f"❌ {load_status}")
    st.stop()

# 提取资源
best_model = resources["best_model"]
raw_data = resources["raw_data"]
detailed_results = resources["detailed_results"]
best_model_name = detailed_results.iloc[0]["模型"]  # 最优模型名称

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="在线索赔预测",
    page_icon="🎯",
    layout="wide"
)

# ---------------------- 主内容 ----------------------
st.title("🎯 在线医疗保险索赔金额预测")
st.divider()

# 预测方式选择
prediction_mode = st.radio(
    "选择预测方式",
    ["📝 单条数据预测", "📁 批量CSV文件预测"],
    horizontal=True
)

if prediction_mode == "📝 单条数据预测":
    st.markdown("请填写以下个人与保险信息，系统将基于最优模型预测索赔金额")

    # 1. 预测表单
    with st.form("prediction_form", clear_on_submit=True):
        # 分三列布局输入框
        col1, col2, col3 = st.columns(3)
        # 左列：基础信息
        with col1:
            st.markdown("### 基础信息")
            age = st.number_input("年龄", min_value=0, max_value=100, value=45, step=1)
            sex = st.selectbox("性别", ["Male", "Female"], index=0)
            smoker = st.selectbox("吸烟状态", ["Never", "Former", "Current"], index=0)
            bmi = st.number_input("BMI指数", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        # 中列：健康信息
        with col2:
            st.markdown("### 健康信息")
            systolic_bp = st.number_input("收缩压", min_value=80, max_value=200, value=120, step=1)
            diastolic_bp = st.number_input("舒张压", min_value=50, max_value=120, value=80, step=1)
            chronic_count = st.number_input("慢性病数量", min_value=0, max_value=10, value=1, step=1)
            hypertension = st.selectbox("高血压", ["Yes", "No"], index=1)
            diabetes = st.selectbox("糖尿病", ["Yes", "No"], index=1)
        # 右列：保险与医疗信息
        with col3:
            st.markdown("### 保险与医疗信息")
            income = st.number_input("年收入（元）", min_value=0, max_value=500000, value=100000, step=1000)
            plan_type = st.selectbox("保险计划类型", ["Basic", "Standard", "Premium"], index=1)
            network_tier = st.selectbox("网络等级", ["Tier 1", "Tier 2", "Tier 3"], index=0)
            deductible = st.number_input("自付额（元）", min_value=0, max_value=50000, value=5000, step=100)
        # 第二行输入框
        col4, col5 = st.columns(2)
        with col4:
            annual_premium = st.number_input("年度保费（元）", min_value=0, max_value=50000, value=8000, step=100)
            visits_last_year = st.number_input("去年就诊次数", min_value=0, max_value=50, value=3, step=1)
        with col5:
            medication_count = st.number_input("用药种类数", min_value=0, max_value=30, value=5, step=1)
        # 提交按钮
        submit_btn = st.form_submit_button("🚀 开始预测", type="primary")

    # 2. 预测逻辑（含新增的MySQL写入）
    if submit_btn:
        # 构建输入数据框（与训练时特征一致）
        input_data = pd.DataFrame({
            "age": [age], "sex": [sex], "smoker": [smoker], "bmi": [bmi],
            "systolic_bp": [systolic_bp], "diastolic_bp": [diastolic_bp],
            "chronic_count": [chronic_count], "income": [income],
            "plan_type": [plan_type], "network_tier": [network_tier],
            "deductible": [deductible], "annual_premium": [annual_premium],
            "visits_last_year": [visits_last_year], "medication_count": [medication_count],
            "hypertension": [hypertension], "diabetes": [diabetes],
            "risk_score": [5.0], "copay": [1000], "claims_count": [1], "annual_medical_cost": [10000]
        })

        # 特征工程（与训练时保持一致）
        input_data = quick_feature_engineering(input_data)

        # 计算零值比例（用于预测值逆变换）
        if raw_data is not None:
            zero_ratio = (raw_data["total_claims_paid"] == 0).mean()
        else:
            zero_ratio = 0.2

        # 执行预测与结果处理
        with st.spinner("🔄 正在计算预测结果..."):
            try:
                # 模型预测（对数变换后的结果）
                y_pred_transformed = best_model.predict(input_data)[0]

                # 预测值逆变换（还原为原始索赔金额）
                if zero_ratio > 0.3:
                    predicted_claims = np.expm1(y_pred_transformed)
                else:
                    predicted_claims = np.expm1(y_pred_transformed)

                # ---------------------- 新增：调用MySQL写入函数 ----------------------
                # 获取最优模型的R²得分（用于存入数据库）
                best_model_r2 = detailed_results.iloc[0]["R²"]
                # 存入MySQL（返回是否成功）
                save_success = save_prediction_to_mysql(
                    input_data=input_data,
                    predicted_claims=predicted_claims,
                    model_name=best_model_name,
                    r2_score=best_model_r2
                )

                # 显示预测结果（含MySQL保存状态）
                st.success(
                    "✅ 预测完成！" + ("（记录已保存至数据库）" if save_success else "（⚠️ 数据库保存失败，请检查配置）"))
                st.divider()

                # 结果展示（原有逻辑不变）
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.info("📋 输入信息摘要")
                    st.write(f"年龄：{age} 岁 | 性别：{sex} | 吸烟状态：{smoker}")
                    st.write(f"BMI：{bmi} | 慢性病数量：{chronic_count} 种")
                    st.write(f"高血压：{hypertension} | 糖尿病：{diabetes}")
                    st.write(f"保险计划：{plan_type} | 年度保费：{annual_premium:,} 元")
                with res_col2:
                    st.success("💰 索赔金额预测结果")
                    st.metric(
                        label="预测索赔金额",
                        value=f"¥{predicted_claims:,.2f}",
                        delta=f"基于「{best_model_name}」预测"
                    )
                    st.write(f"模型R²得分：{best_model_r2:.4f}（解释力）")
                    st.write(f"模型RMSE：{detailed_results.iloc[0]['RMSE']:.2f}（平均误差）")

            except Exception as e:
                st.error(f"❌ 预测失败：{str(e)}")

else:  # 批量CSV文件预测
    st.markdown("""
    ### 📁 批量CSV文件预测

    请上传包含以下字段的CSV文件进行批量预测：

    **必需字段**：
    - `age` (年龄)
    - `sex` (性别: Male/Female)
    - `smoker` (吸烟状态: Never/Former/Current)
    - `bmi` (BMI指数)
    - `systolic_bp` (收缩压)
    - `diastolic_bp` (舒张压)
    - `chronic_count` (慢性病数量)
    - `income` (年收入)
    - `plan_type` (保险计划类型: Basic/Standard/Premium)
    - `network_tier` (网络等级: Tier 1/Tier 2/Tier 3)
    - `deductible` (自付额)
    - `annual_premium` (年度保费)
    - `visits_last_year` (去年就诊次数)
    - `medication_count` (用药种类数)
    - `hypertension` (高血压: Yes/No)
    - `diabetes` (糖尿病: Yes/No)

    **可选字段**（如缺失将使用默认值）：
    - `risk_score` (风险评分，默认5.0)
    - `copay` (共付额，默认1000)
    - `claims_count` (索赔次数，默认1)
    - `annual_medical_cost` (年度医疗成本，默认10000)
    """)

    # 文件上传组件
    uploaded_file = st.file_uploader(
        "选择CSV文件",
        type=["csv"],
        help="请确保CSV文件包含必需的字段，且编码为UTF-8"
    )

    if uploaded_file is not None:
        try:
            # 读取上传的CSV文件
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ 成功读取文件，共 {len(batch_data)} 条记录")

            # 显示数据预览
            st.subheader("📋 上传数据预览")
            st.dataframe(batch_data.head(10), use_container_width=True)

            # 检查必需字段
            required_columns = [
                'age', 'sex', 'smoker', 'bmi', 'systolic_bp', 'diastolic_bp',
                'chronic_count', 'income', 'plan_type', 'network_tier', 'deductible',
                'annual_premium', 'visits_last_year', 'medication_count',
                'hypertension', 'diabetes'
            ]

            missing_columns = [col for col in required_columns if col not in batch_data.columns]

            if missing_columns:
                st.error(f"❌ CSV文件缺少以下必需字段：{', '.join(missing_columns)}")
                st.stop()

            # 添加默认值字段（如果缺失）
            optional_columns_defaults = {
                'risk_score': 5.0,
                'copay': 1000,
                'claims_count': 1,
                'annual_medical_cost': 10000
            }

            for col, default_val in optional_columns_defaults.items():
                if col not in batch_data.columns:
                    batch_data[col] = default_val
                    st.info(f"ℹ️ 已为缺失字段 `{col}` 添加默认值: {default_val}")

            # 开始批量预测
            if st.button("🚀 开始批量预测", type="primary"):
                with st.spinner(f"🔄 正在批量预测 {len(batch_data)} 条记录..."):
                    try:
                        # 特征工程
                        batch_data_processed = quick_feature_engineering(batch_data)

                        # 计算零值比例（用于预测值逆变换）
                        if raw_data is not None:
                            zero_ratio = (raw_data["total_claims_paid"] == 0).mean()
                        else:
                            zero_ratio = 0.2

                        # 批量预测
                        y_pred_transformed = best_model.predict(batch_data_processed)

                        # 预测值逆变换
                        if zero_ratio > 0.3:
                            predicted_claims = np.expm1(y_pred_transformed)
                        else:
                            predicted_claims = np.expm1(y_pred_transformed)

                        # 添加预测结果到数据
                        result_data = batch_data.copy()
                        result_data['predicted_claims'] = predicted_claims
                        result_data['predicted_claims'] = result_data['predicted_claims'].round(2)
                        result_data['model_name'] = best_model_name
                        result_data['r2_score'] = detailed_results.iloc[0]["R²"]

                        # 显示预测结果
                        st.success(f"✅ 批量预测完成！共处理 {len(result_data)} 条记录")

                        # 结果预览
                        st.subheader("📊 批量预测结果预览")
                        display_columns = required_columns + ['predicted_claims']
                        st.dataframe(result_data[display_columns].head(15), use_container_width=True)

                        # 统计信息
                        st.subheader("📈 预测结果统计")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("总记录数", f"{len(result_data):,} 条")
                        with col2:
                            st.metric("平均预测金额", f"¥{result_data['predicted_claims'].mean():,.2f}")
                        with col3:
                            st.metric("最小预测金额", f"¥{result_data['predicted_claims'].min():,.2f}")
                        with col4:
                            st.metric("最大预测金额", f"¥{result_data['predicted_claims'].max():,.2f}")

                        # 下载预测结果
                        st.subheader("💾 下载预测结果")
                        csv_data = result_data.to_csv(index=False, encoding="utf-8-sig")

                        st.download_button(
                            label="📥 下载完整预测结果CSV",
                            data=csv_data,
                            file_name=f"批量预测结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            type="primary"
                        )

                        # 批量保存到MySQL的选项
                        st.subheader("🗄️ 数据库保存选项")
                        if st.checkbox("将批量预测结果保存到数据库"):
                            with st.spinner("正在保存到数据库..."):
                                save_count = 0
                                for i, row in result_data.iterrows():
                                    # 构建单条输入数据
                                    single_input = pd.DataFrame(
                                        {row.index[j]: [row.values[j]] for j in range(len(row))})
                                    single_input = single_input[
                                        required_columns + list(optional_columns_defaults.keys())]

                                    save_success = save_prediction_to_mysql(
                                        input_data=single_input,
                                        predicted_claims=row['predicted_claims'],
                                        model_name=best_model_name,
                                        r2_score=detailed_results.iloc[0]["R²"]
                                    )
                                    if save_success:
                                        save_count += 1

                                if save_count > 0:
                                    st.success(f"✅ 成功保存 {save_count}/{len(result_data)} 条记录到数据库")
                                else:
                                    st.error("❌ 数据库保存失败，请检查MySQL配置")

                    except Exception as e:
                        st.error(f"❌ 批量预测失败：{str(e)}")

        except Exception as e:
            st.error(f"❌ 文件读取失败：{str(e)}")
            st.info("💡 请确保上传的文件是有效的CSV格式，且编码为UTF-8")