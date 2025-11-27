# 创建一个更完整的测试数据生成脚本
import pandas as pd
import numpy as np


def create_test_csv_with_all_fields():
    """创建包含所有必需和可选字段的测试CSV"""
    np.random.seed(42)
    num_records = 50

    # 创建包含所有字段的数据
    test_data = pd.DataFrame({
        # 必需字段
        'age': np.random.randint(18, 80, num_records),
        'sex': np.random.choice(['Male', 'Female'], num_records, p=[0.6, 0.4]),
        'smoker': np.random.choice(['Never', 'Former', 'Current'], num_records, p=[0.6, 0.2, 0.2]),
        'bmi': np.round(np.random.normal(25, 5, num_records), 1),
        'systolic_bp': np.random.randint(100, 180, num_records),
        'diastolic_bp': np.random.randint(60, 110, num_records),
        'chronic_count': np.random.randint(0, 5, num_records),
        'income': np.random.randint(30000, 150000, num_records),
        'plan_type': np.random.choice(['Basic', 'Standard', 'Premium'], num_records, p=[0.3, 0.5, 0.2]),
        'network_tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], num_records, p=[0.4, 0.4, 0.2]),
        'deductible': np.random.choice([1000, 2000, 3000, 5000], num_records, p=[0.2, 0.3, 0.3, 0.2]),
        'annual_premium': np.random.randint(2000, 15000, num_records),
        'visits_last_year': np.random.randint(0, 20, num_records),
        'medication_count': np.random.randint(0, 10, num_records),
        'hypertension': np.random.choice(['Yes', 'No'], num_records, p=[0.3, 0.7]),
        'diabetes': np.random.choice(['Yes', 'No'], num_records, p=[0.2, 0.8]),

        # 可选字段（避免系统自动添加）
        'risk_score': np.round(np.random.uniform(1, 10, num_records), 1),
        'copay': np.random.choice([500, 1000, 1500, 2000], num_records),
        'claims_count': np.random.randint(0, 5, num_records),
        'annual_medical_cost': np.random.randint(5000, 50000, num_records)
    })

    # 确保数据合理性
    test_data['bmi'] = test_data['bmi'].clip(15, 40)

    # 保存文件
    filename = '完整字段测试数据.csv'
    test_data.to_csv(filename, index=False, encoding='utf-8-sig')

    print(f"✅ 完整测试CSV已生成: '{filename}'")
    print(f"📊 记录数量: {len(test_data)}")
    print(f"📋 字段数量: {len(test_data.columns)}")
    print("\n字段列表:")
    for col in test_data.columns:
        print(f"  - {col}")

    return test_data


# 生成测试文件
test_data = create_test_csv_with_all_fields()
print("\n📈 数据预览:")
print(test_data.head())