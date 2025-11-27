import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import joblib
import time

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 添加学习曲线绘制函数
def plot_learning_curve(estimator, title, X, y, cv=3, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    绘制学习曲线
    """
    plt.figure(figsize=(10, 6))

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes,
        scoring='r2', n_jobs=1, random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练集得分")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="验证集得分")

    plt.xlabel("训练样本数")
    plt.ylabel("R²得分")
    plt.title(f"{title} - 学习曲线")
    plt.legend(loc="best")
    plt.grid(True)

    # 保存图片
    plt.tight_layout()
    plt.savefig(f"{title}_学习曲线.png", dpi=150, bbox_inches='tight')
    plt.close()

    return train_scores_mean, test_scores_mean


# ---------------------- 1. 数据加载与基础清洗 ----------------------
print("=" * 50)
print("Step 1/4: 数据加载与基础清洗...")
start_time = time.time()

# 读取数据
df_raw = pd.read_csv("medical_insurance.csv")
print(f"数据形状：{df_raw.shape}")

# 目标变量
target_col = "total_claims_paid"
print(f"目标变量：{target_col}")

# 基础清洗 - 简化版
df_raw = df_raw.drop_duplicates()
constant_cols = [col for col in df_raw.columns if df_raw[col].nunique() == 1]
df_raw = df_raw.drop(constant_cols, axis=1)
print(f"删除的常数特征：{constant_cols}")

# 目标变量分析
zero_ratio = (df_raw[target_col] == 0).mean()
print(f"零值比例: {zero_ratio:.2%}")

# 目标变量变换
if zero_ratio > 0.3:
    df_raw[f"transformed_{target_col}"] = np.where(
        df_raw[target_col] > 0,
        np.log1p(df_raw[target_col]),
        0
    )
    print("使用两阶段方法处理零膨胀数据")
else:
    df_raw[f"transformed_{target_col}"] = np.log1p(df_raw[target_col])
    print("使用对数变换")

target_col_transformed = f"transformed_{target_col}"

print(f"Step 1 完成！耗时：{time.time() - start_time:.2f}秒\n")

# ---------------------- 2. 快速特征工程 ----------------------
print("Step 2/4: 快速特征工程...")
start_time = time.time()

# 数据准备 - 只删除明显无关的特征
drop_cols = ["person_id", "monthly_premium"]
X = df_raw.drop([target_col, target_col_transformed] + drop_cols, axis=1)
y = df_raw[target_col_transformed]

print(f"特征数量: {X.shape[1]}, 目标变量: {target_col_transformed}")

# 只选择最重要的特征，减少维度
important_numeric_features = [
    "age", "income", "bmi", "risk_score", "chronic_count", "deductible",
    "copay", "visits_last_year", "medication_count", "annual_medical_cost",
    "annual_premium", "claims_count", "systolic_bp", "diastolic_bp"
]
numeric_features = [col for col in important_numeric_features if col in X.columns]

important_categorical_features = [
    "sex", "smoker", "plan_type", "network_tier", "hypertension", "diabetes"
]
categorical_features = [col for col in important_categorical_features if col in X.columns]

# 快速处理缺失值
for col in numeric_features:
    if col in X.columns and X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

for col in categorical_features:
    if col in X.columns and X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else "Unknown")
    if col in X.columns:
        X[col] = X[col].astype(str)

print(f"数值特征：{len(numeric_features)}个")
print(f"类别特征：{len(categorical_features)}个")


# 快速特征工程 - 只添加最重要的几个特征
def quick_feature_engineering(data):
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


X = quick_feature_engineering(X)

# 更新特征列表
new_numeric_features = ["basic_health_risk", "smoker_encoded", "is_senior", "premium_ratio"]
final_numeric_features = numeric_features + new_numeric_features
final_numeric_features = [col for col in final_numeric_features if col in X.columns]

final_categorical_features = categorical_features

print(f"最终数值特征：{len(final_numeric_features)}个")
print(f"最终类别特征：{len(final_categorical_features)}个")

print(f"Step 2 完成！耗时：{time.time() - start_time:.2f}秒\n")

# ---------------------- 3. 扩展模型训练 ----------------------
print("Step 3/4: 扩展模型训练...")
start_time = time.time()

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"训练集：{X_train.shape}，测试集：{X_test.shape}")

# 扩展模型集合 - 包含更多类型的模型
models = {
    # 集成学习模型
    "随机森林回归": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "param_dist": {
            "n_estimators": [100, 150, 200],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [2, 4, 6],
            "max_features": [0.6, 0.8, 1.0]
        }
    },
    "XGBoost回归": {
        "model": XGBRegressor(random_state=42, objective="reg:squarederror", n_jobs=-1),
        "param_dist": {
            "learning_rate": [0.05, 0.1, 0.15],
            "n_estimators": [100, 150, 200],
            "max_depth": [6, 8, 10],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0]
        }
    },
    "LightGBM回归": {
        "model": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "param_dist": {
            "learning_rate": [0.05, 0.1, 0.15],
            "n_estimators": [100, 150, 200],
            "max_depth": [6, 8, 10],
            "num_leaves": [31, 50, 70],
            "subsample": [0.8, 0.9, 1.0]
        }
    },
    "梯度提升回归": {
        "model": GradientBoostingRegressor(random_state=42),
        "param_dist": {
            "learning_rate": [0.05, 0.1, 0.15],
            "n_estimators": [100, 150, 200],
            "max_depth": [3, 4, 5],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [2, 4, 6]
        }
    },
    "极端随机树回归": {
        "model": ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "param_dist": {
            "n_estimators": [100, 150, 200],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [2, 4, 6],
            "max_features": [0.6, 0.8, 1.0]
        }
    },
    # 线性模型
    "岭回归": {
        "model": Ridge(random_state=42),
        "param_dist": {
            "alpha": [0.1, 1.0, 10.0, 100.0],
            "solver": ["auto", "svd", "cholesky", "lsqr"]
        }
    },
    "Lasso回归": {
        "model": Lasso(random_state=42),
        "param_dist": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "selection": ["cyclic", "random"]
        }
    },
    "弹性网络回归": {
        "model": ElasticNet(random_state=42),
        "param_dist": {
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    },
    # 其他模型
    "支持向量回归": {
        "model": SVR(),
        "param_dist": {
            "C": [0.1, 1.0, 10.0],
            "epsilon": [0.01, 0.1, 0.2],
            "kernel": ["linear", "rbf"]
        }
    },
    "K近邻回归": {
        "model": KNeighborsRegressor(),
        "param_dist": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }
    },
    "决策树回归": {
        "model": DecisionTreeRegressor(random_state=42),
        "param_dist": {
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [5, 10, 15, 20],
            "min_samples_leaf": [2, 4, 6, 8]
        }
    },
    "AdaBoost回归": {
        "model": AdaBoostRegressor(random_state=42),
        "param_dist": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.1, 1.0],
            "loss": ["linear", "square", "exponential"]
        }
    }
}

# 训练与评估
best_models = {}
model_scores = {}
preprocessed_predictions = {}

# 首先预处理数据
print("预处理数据...")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), final_numeric_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), final_categorical_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"预处理后训练集形状: {X_train_processed.shape}")
print(f"预处理后测试集形状: {X_test_processed.shape}")

# 为了加速训练，我们可以将模型分为两组：快速模型和慢速模型
fast_models = ["随机森林回归", "XGBoost回归", "LightGBM回归", "梯度提升回归",
               "极端随机树回归", "岭回归", "Lasso回归", "弹性网络回归", "决策树回归", "AdaBoost回归"]
slow_models = ["支持向量回归", "K近邻回归"]

print(f"将训练 {len(fast_models)} 个快速模型和 {len(slow_models)} 个慢速模型")

for name, info in models.items():
    # 可以选择跳过慢速模型以节省时间
    if name in slow_models and len(best_models) >= 5:  # 如果有足够多的好模型，跳过慢速模型
        print(f"跳过 {name}（慢速模型）")
        continue

    model_start = time.time()
    print(f"\n训练 {name}...")

    try:
        # 使用更少的迭代次数和交叉验证折数以加速
        n_iter = 3 if name in slow_models else 4
        cv = 2 if name in slow_models else 3

        random_search = RandomizedSearchCV(
            info["model"], info["param_dist"], n_iter=n_iter, cv=cv, scoring="r2",
            n_jobs=1, verbose=1, random_state=42
        )
        random_search.fit(X_train_processed, y_train)

        y_pred_transformed = random_search.best_estimator_.predict(X_test_processed)

        # 逆变换
        if zero_ratio > 0.3:
            y_pred = np.expm1(y_pred_transformed)
        else:
            y_pred = np.expm1(y_pred_transformed)

        y_test_original = df_raw.loc[y_test.index, target_col]

        r2 = r2_score(y_test_original, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
        mae = mean_absolute_error(y_test_original, y_pred)

        best_models[name] = random_search.best_estimator_
        model_scores[name] = {
            "R²": r2, "RMSE": rmse, "MAE": mae,
            "最优参数": random_search.best_params_
        }

        # 保存预测结果用于集成
        preprocessed_predictions[name] = y_pred_transformed

        print(f"{name} 完成！耗时：{time.time() - model_start:.2f}秒")
        print(f"  - R²：{r2:.4f}，RMSE：{rmse:.2f}，MAE：{mae:.2f}")

        # 只为性能最好的几个模型绘制学习曲线
        if r2 > 0.5 or len(best_models) <= 3:  # 只对性能好的或前几个模型绘制学习曲线
            print(f"  绘制{name}学习曲线...")
            try:
                # 使用部分数据绘制学习曲线以加速
                if X_train_processed.shape[0] > 1000:
                    indices = np.random.choice(X_train_processed.shape[0], 1000, replace=False)
                    X_sample = X_train_processed[indices]
                    y_sample = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
                else:
                    X_sample = X_train_processed
                    y_sample = y_train

                plot_learning_curve(
                    random_search.best_estimator_,
                    name,
                    X_sample,
                    y_sample,
                    cv=2  # 减少CV折数加速
                )
                print(f"  {name}学习曲线已保存")
            except Exception as e:
                print(f"  绘制{name}学习曲线失败: {str(e)}")

    except Exception as e:
        print(f"{name} 训练失败：{str(e)}")
        continue

# 多种集成方法
if len(preprocessed_predictions) >= 2:
    print("\n尝试多种集成方法...")
    ensemble_start = time.time()

    # 获取测试集的原始目标变量
    y_test_original = df_raw.loc[y_test.index, target_col]

    # 1. 简单平均集成
    all_predictions = list(preprocessed_predictions.values())
    y_pred_transformed_simple = np.mean(all_predictions, axis=0)

    # 2. 加权平均集成（基于R²分数）
    r2_scores = [model_scores[name]["R²"] for name in preprocessed_predictions.keys()]
    weights = [max(0, score) for score in r2_scores]  # 确保权重非负
    if sum(weights) > 0:
        weights = [w / sum(weights) for w in weights]
        y_pred_transformed_weighted = np.average(all_predictions, axis=0, weights=weights)
    else:
        y_pred_transformed_weighted = y_pred_transformed_simple

    # 3. 中位数集成（对异常值更鲁棒）
    y_pred_transformed_median = np.median(all_predictions, axis=0)

    # 逆变换并评估各种集成方法
    ensemble_methods = {
        "简单平均集成": y_pred_transformed_simple,
        "加权平均集成": y_pred_transformed_weighted,
        "中位数集成": y_pred_transformed_median
    }

    for method_name, y_pred_transformed in ensemble_methods.items():
        # 逆变换
        if zero_ratio > 0.3:
            y_pred_ensemble = np.expm1(y_pred_transformed)
        else:
            y_pred_ensemble = np.expm1(y_pred_transformed)

        r2_ensemble = r2_score(y_test_original, y_pred_ensemble)
        rmse_ensemble = np.sqrt(mean_squared_error(y_test_original, y_pred_ensemble))
        mae_ensemble = mean_absolute_error(y_test_original, y_pred_ensemble)

        model_scores[method_name] = {
            "R²": r2_ensemble, "RMSE": rmse_ensemble, "MAE": mae_ensemble,
            "最优参数": method_name
        }

        print(f"{method_name} 完成！")
        print(f"  - R²：{r2_ensemble:.4f}，RMSE：{rmse_ensemble:.2f}，MAE：{mae_ensemble:.2f}")

    print(f"所有集成方法完成！耗时：{time.time() - ensemble_start:.2f}秒")

# 创建完整的pipeline用于保存最佳模型
if best_models:
    # 找到最佳模型（排除集成方法）
    individual_models = {name: score for name, score in model_scores.items()
                         if "集成" not in name and name in best_models}
    if individual_models:
        best_model_name = max(individual_models.keys(), key=lambda x: model_scores[x]["R²"])
        best_model = best_models[best_model_name]

        # 创建完整的pipeline
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', best_model)
        ])
        # 重新拟合完整pipeline
        full_pipeline.fit(X_train, y_train)
        best_models[best_model_name] = full_pipeline

        print(f"\n=== 最优个体模型：{best_model_name} ===")
        print(f"最优性能：R²={model_scores[best_model_name]['R²']:.4f}")

    # 找到最佳模型（包括集成方法）
    best_overall_name = max(model_scores.keys(), key=lambda x: model_scores[x]["R²"])
    print(f"\n=== 最优模型（包括集成）：{best_overall_name} ===")
    print(f"最优性能：R²={model_scores[best_overall_name]['R²']:.4f}")

print(f"Step 3 完成！耗时：{time.time() - start_time:.2f}秒\n")

# ---------------------- 4. 详细结果分析 ----------------------
print("Step 4/4: 详细结果分析...")
start_time = time.time()

if model_scores:
    # 生成详细的性能对比图
    plt.figure(figsize=(14, 8))
    model_names = list(model_scores.keys())
    r2_scores = [model_scores[name]["R²"] for name in model_names]

    # 为不同类型的模型设置不同颜色
    colors = []
    for name in model_names:
        if "集成" in name:
            colors.append("purple")  # 集成方法用紫色
        elif name in ["岭回归", "Lasso回归", "弹性网络回归"]:
            colors.append("orange")  # 线性模型用橙色
        elif name in ["支持向量回归", "K近邻回归"]:
            colors.append("brown")  # 其他模型用棕色
        else:
            colors.append("skyblue")  # 树模型用蓝色

    # 标记最佳模型
    best_idx = np.argmax(r2_scores)
    colors[best_idx] = "red"

    bars = plt.bar(range(len(model_names)), r2_scores, color=colors, alpha=0.8)

    plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right")
    plt.title("扩展模型性能对比（红色最优）", fontsize=14, fontweight='bold')
    plt.ylabel("R²（越高越好）")
    plt.grid(axis="y", alpha=0.3)

    # 添加数值标签
    for i, (bar, r2) in enumerate(zip(bars, r2_scores)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{r2:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 添加图例
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='skyblue', label='树模型/集成模型'),
        Patch(facecolor='orange', label='线性模型'),
        Patch(facecolor='brown', label='其他模型'),
        Patch(facecolor='purple', label='集成方法'),
        Patch(facecolor='red', label='最优模型')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig("扩展模型性能对比.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 保存详细结果
    results_df = pd.DataFrame([
        {
            "模型": name,
            "R²": scores["R²"],
            "RMSE": scores["RMSE"],
            "MAE": scores["MAE"],
            "类型": "集成方法" if "集成" in name else
            "线性模型" if name in ["岭回归", "Lasso回归", "弹性网络回归"] else
            "其他模型" if name in ["支持向量回归", "K近邻回归"] else "树模型/集成模型"
        }
        for name, scores in model_scores.items()
    ]).sort_values("R²", ascending=False)

    results_df.to_csv("模型性能详细结果.csv", index=False, encoding='utf-8-sig')

    # 保存模型和预处理器
    if best_overall_name in best_models and "集成" not in best_overall_name:
        joblib.dump(best_models[best_overall_name], "最优模型.pkl")
        print("已保存最优模型")
    joblib.dump(model_scores, "模型性能.pkl")
    joblib.dump(preprocessor, "预处理器.pkl")

    # 输出详细总结
    print(f"\n=== 最终结果详细总结 ===")
    print(f"总共训练了 {len(model_scores)} 个模型")
    print("\n性能排名前5的模型:")
    top_5 = results_df.head(5)
    for _, row in top_5.iterrows():
        print(f"{row['模型']}: R²={row['R²']:.4f}, RMSE={row['RMSE']:.2f}, MAE={row['MAE']:.2f}")

    print(f"\n最佳模型: {best_overall_name}")
    print(f"最佳R²: {model_scores[best_overall_name]['R²']:.4f}")
    print(f"最佳RMSE: {model_scores[best_overall_name]['RMSE']:.2f}")
    print(f"最佳MAE: {model_scores[best_overall_name]['MAE']:.2f}")

print(f"Step 4 完成！耗时：{time.time() - start_time:.2f}秒\n")
print("=" * 50)
print("扩展模型训练脚本执行完成！")