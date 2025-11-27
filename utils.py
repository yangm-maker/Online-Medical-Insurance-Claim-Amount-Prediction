# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import StandardScaler
import warnings
# ---------------------- 新增：MySQL 依赖库 ----------------------
import pymysql
from sqlalchemy import create_engine
from datetime import datetime

warnings.filterwarnings('ignore')


# ---------------------- 全局配置 ----------------------
def set_chinese_font():
    """设置中文字体，避免乱码"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


# ---------------------- 资源加载 ----------------------
@st.cache_resource(show_spinner="正在加载模型和数据...")
def load_all_resources():
    """加载所有训练好的资源（模型、预处理器、性能数据）"""
    resources = {}
    try:
        # 核心模型文件
        resources["best_model"] = joblib.load("最优模型.pkl")
        resources["preprocessor"] = joblib.load("预处理器.pkl")
        resources["model_scores"] = joblib.load("模型性能.pkl")
        # 辅助数据文件（可选）
        resources["raw_data"] = pd.read_csv("medical_insurance.csv") if os.path.exists(
            "medical_insurance.csv") else None
        resources["detailed_results"] = pd.read_csv("模型性能详细结果.csv", encoding='utf-8-sig') if os.path.exists(
            "模型性能详细结果.csv") else None
        # 生成详细结果（若CSV不存在则从model_scores构建）
        if resources["detailed_results"] is None:
            results_data = []
            for model_name, scores in resources["model_scores"].items():
                model_type = _get_model_type(model_name)
                results_data.append({
                    "模型": model_name,
                    "R²": scores["R²"],
                    "RMSE": scores["RMSE"],
                    "MAE": scores["MAE"],
                    "类型": model_type
                })
            resources["detailed_results"] = pd.DataFrame(results_data).sort_values("R²", ascending=False)
        return resources, "success"
    except FileNotFoundError as e:
        return None, f"缺少文件：{str(e)}"
    except Exception as e:
        return None, f"加载失败：{str(e)}"


# ---------------------- 辅助函数 ----------------------
def _get_model_type(model_name):
    """判断模型类型（用于分类显示）"""
    if "集成" in model_name:
        return "集成方法"
    elif model_name in ["岭回归", "Lasso回归", "弹性网络回归"]:
        return "线性模型"
    elif model_name in ["支持向量回归", "K近邻回归"]:
        return "其他模型"
    else:
        return "树模型/集成模型"


def quick_feature_engineering(data):
    """特征工程（与训练时保持一致）"""
    df_new = data.copy()
    # 1. 基础风险评分
    if 'chronic_count' in df_new.columns:
        df_new["basic_health_risk"] = df_new["chronic_count"] * 2
    # 2. 吸烟状态编码
    if 'smoker' in df_new.columns:
        smoker_map = {"Never": 0, "Former": 1, "Current": 2}
        df_new["smoker_encoded"] = df_new["smoker"].map(lambda x: smoker_map.get(x, 0))
    # 3. 年龄分组
    if 'age' in df_new.columns:
        df_new["is_senior"] = (df_new["age"] >= 65).astype(int)
    # 4. 保险费用比例
    if all(col in df_new.columns for col in ["annual_premium", "deductible"]):
        df_new["premium_ratio"] = df_new["annual_premium"] / (df_new["deductible"] + 100)
    return df_new


def get_learning_curve_files():
    """获取所有学习曲线图片文件"""
    curve_files = []
    for file in os.listdir("."):
        if file.endswith("_学习曲线.png"):
            curve_files.append(file)
    return curve_files


# ---------------------- 新增：MySQL 相关函数 ----------------------
# TODO：替换为你的 MySQL 实际配置（必须修改！）
MYSQL_CONFIG = {
    "host": "localhost",  # 本地MySQL填"localhost"，云服务器填公网IP
    "user": "root",  # 数据库用户名（如root）
    "password": "123456",  # 你的MySQL密码（如123456）
    "database": "insurance_prediction",  # 数据库名（需提前创建）
    "charset": "utf8mb4"  # 字符编码（避免中文乱码）
}


def save_prediction_to_mysql(input_data, predicted_claims, model_name, r2_score):
    """
    将在线预测结果存入MySQL
    参数：
        input_data: 用户输入的特征数据（DataFrame格式）
        predicted_claims: 模型预测的索赔金额（数值）
        model_name: 所用最优模型名称（字符串）
        r2_score: 最优模型的R²得分（数值）
    返回：
        bool: 成功返回True，失败返回False
    """
    try:
        # 1. 创建MySQL连接引擎（兼容Pandas to_sql方法）
        conn_str = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}?charset={MYSQL_CONFIG['charset']}"
        engine = create_engine(conn_str)

        # 2. 提取用户输入特征（与prediction_records表字段一一对应）
        input_features = {
            "age": input_data["age"].iloc[0],
            "sex": input_data["sex"].iloc[0],
            "smoker": input_data["smoker"].iloc[0],
            "bmi": input_data["bmi"].iloc[0],
            "systolic_bp": input_data["systolic_bp"].iloc[0],
            "diastolic_bp": input_data["diastolic_bp"].iloc[0],
            "chronic_count": input_data["chronic_count"].iloc[0],
            "hypertension": input_data["hypertension"].iloc[0],
            "diabetes": input_data["diabetes"].iloc[0],
            "income": input_data["income"].iloc[0],
            "plan_type": input_data["plan_type"].iloc[0],
            "network_tier": input_data["network_tier"].iloc[0],
            "deductible": input_data["deductible"].iloc[0],
            "annual_premium": input_data["annual_premium"].iloc[0],
            "visits_last_year": input_data["visits_last_year"].iloc[0],
            "medication_count": input_data["medication_count"].iloc[0]
        }

        # 3. 拼接预测结果与模型信息
        prediction_data = {
            **input_features,
            "predicted_claims": round(predicted_claims, 2),  # 保留2位小数
            "model_name": model_name,
            "r2_score": round(r2_score, 4)  # 保留4位小数
        }

        # 4. 转换为DataFrame并写入MySQL（追加模式）
        df_to_save = pd.DataFrame([prediction_data])
        df_to_save.to_sql(
            name="prediction_records",  # 目标表名（需提前创建）
            con=engine,
            if_exists="append",  # 追加记录，不覆盖
            index=False,  # 不写入索引
            chunksize=10
        )
        return True
    except Exception as e:
        print(f"MySQL写入失败：{str(e)}")  # 控制台打印错误（调试用）
        return False