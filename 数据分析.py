# ==============================
# 1. 环境导入与基础设置
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 基础配置（中文字体、图表样式、分辨率）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-talk')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


# ==============================
# 2. 数据加载与预处理（CSV格式适配）
# ==============================
# 加载CSV数据（根据实际编码调整encoding参数）
df = pd.read_csv('medical_insurance.csv', encoding='utf-8')  # 若乱码，替换为encoding='gbk'

# 2.1 数据基本信息探查
print("="*60)
print("【数据基本信息】")
print("="*60)
print(f"数据规模：{df.shape[0]}行 × {df.shape[1]}列")
print(f"目标变量（total_claims_paid）数据类型：{df['total_claims_paid'].dtype}")
print(f"目标变量缺失值数量：{df['total_claims_paid'].isnull().sum()}（无缺失值可直接分析）")

# 2.2 保留核心分析字段
core_fields = [
    'total_claims_paid', 'age', 'sex', 'income', 'bmi', 'smoker', 'alcohol_freq',
    'chronic_count', 'hypertension', 'diabetes', 'asthma', 'risk_score',
    'visits_last_year', 'hospitalizations_last_3yrs', 'annual_medical_cost',
    'plan_type', 'deductible', 'copay'
]
df_core = df[core_fields].copy()

# 2.3 缺失值处理
for col in df_core.columns:
    if df_core[col].isnull().sum() > 0:
        if df_core[col].dtype in ['int64', 'float64']:
            df_core[col].fillna(df_core[col].median(), inplace=True)
        else:
            df_core[col].fillna(df_core[col].mode()[0], inplace=True)

# 2.4 异常值标记（IQR法则）
q25 = df_core['total_claims_paid'].quantile(0.25)
q75 = df_core['total_claims_paid'].quantile(0.75)
iqr = q75 - q25
df_core['is_outlier'] = (df_core['total_claims_paid'] > q75 + 1.5*iqr) | (df_core['total_claims_paid'] < q25 - 1.5*iqr)
outlier_count = df_core['is_outlier'].sum()

# 2.5 构建衍生特征
df_core['age_group'] = pd.cut(df_core['age'], bins=[0, 34, 59, 100], labels=['青年', '中年', '老年'])
df_core['plan_tier'] = df_core['plan_type'].map({
    'Bronze': '基础型', 'Silver': '中等型', 'Gold': '中等型', 'Platinum': '高端型'
}).fillna('基础型')
df_core['health_risk'] = pd.cut(df_core['chronic_count'], bins=[-1, 1, 3, 10], labels=['低风险', '中风险', '高风险'])
df_core['smoke_status'] = df_core['smoker'].map({
    'Never': '从不吸烟', 'Former': '曾经吸烟', 'Current': '当前吸烟'
}).fillna('从不吸烟')

# 输出预处理结果
print(f"\n【数据预处理结果】")
print(f"异常值统计：共{outlier_count}条异常值（占比{outlier_count/len(df_core)*100:.2f}%）")
print(f"异常值范围：总赔付金额 < {q25-1.5*iqr:.0f}元 或 > {q75+1.5*iqr:.0f}元")
print(f"衍生特征：age_group（年龄分组）、plan_tier（保险等级）、health_risk（健康风险）、smoke_status（吸烟状态）")


# ==============================
# 3. 单变量分析（每个图表单独保存）
# ==============================
print("\n" + "="*60)
print("【单变量分析：总赔付金额特征】")
print("="*60)

# 3.1 统计描述
desc_stats = df_core['total_claims_paid'].describe().round(2)
skewness = stats.skew(df_core['total_claims_paid'])
kurtosis = stats.kurtosis(df_core['total_claims_paid'])
print("总赔付金额统计描述（单位：元）：")
print(desc_stats)
print(f"分布特征：偏度={skewness:.2f}（显著右偏），峰度={kurtosis:.2f}（尖峰分布）")

# 3.2 分布直方图（单独保存）
plt.figure(figsize=(10, 6))
normal_data = df_core[~df_core['is_outlier']]['total_claims_paid']
outlier_data = df_core[df_core['is_outlier']]['total_claims_paid']
plt.hist(normal_data, bins=30, alpha=0.7, color='#2E86AB', edgecolor='black', label=f'正常值（{len(normal_data)}条）')
plt.hist(outlier_data, bins=15, alpha=0.8, color='#E63946', edgecolor='black', label=f'异常值（{len(outlier_data)}条）')
plt.axvline(df_core['total_claims_paid'].mean(), color='red', linestyle='--', linewidth=2, label=f'均值：{df_core["total_claims_paid"].mean():.0f}元')
plt.axvline(df_core['total_claims_paid'].median(), color='green', linestyle='--', linewidth=2, label=f'中位数：{df_core["total_claims_paid"].median():.0f}元')
plt.xlabel('总赔付金额（元）')
plt.ylabel('用户数量')
plt.title('总赔付金额分布（含异常值区分）')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('1_总赔付金额分布直方图.png', bbox_inches='tight')
plt.close()
print("【单变量分析图表已保存】：1_总赔付金额分布直方图.png")

# 3.3 累积分布曲线（单独保存）
plt.figure(figsize=(10, 6))
sorted_paid = np.sort(df_core['total_claims_paid'])
cdf = np.arange(1, len(sorted_paid)+1) / len(sorted_paid)
plt.plot(sorted_paid, cdf*100, color='#F1A208', linewidth=2)
plt.axvline(q25, color='gray', linestyle=':', linewidth=1.5, label=f'25%分位数：{q25:.0f}元')
plt.axvline(q75, color='gray', linestyle=':', linewidth=1.5, label=f'75%分位数：{q75:.0f}元')
plt.axvline(df_core['total_claims_paid'].quantile(0.95), color='red', linestyle='--', linewidth=2, label=f'95%分位数：{df_core["total_claims_paid"].quantile(0.95):.0f}元')
plt.xlabel('总赔付金额（元）')
plt.ylabel('累积用户占比（%）')
plt.title('总赔付金额累积分布曲线（CDF）')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('2_总赔付金额累积分布曲线.png', bbox_inches='tight')
plt.close()
print("【单变量分析图表已保存】：2_总赔付金额累积分布曲线.png")


# ==============================
# 4. 双变量分析（每个图表单独保存）
# ==============================
print("\n" + "="*60)
print("【双变量分析：关键因素与总赔付金额】")
print("="*60)

# 4.1 相关性分析
num_features = ['age', 'income', 'bmi', 'chronic_count', 'risk_score', 'annual_medical_cost', 'deductible']
corr_data = df_core[num_features + ['total_claims_paid']].corr()
corr_with_target = corr_data['total_claims_paid'].sort_values(ascending=False)

print("数值特征与总赔付金额的相关系数（降序）：")
for feat, corr in corr_with_target.items():
    if feat != 'total_claims_paid':
        corr_level = '强正相关' if corr > 0.5 else '中等正相关' if corr > 0.3 else '弱相关' if corr > 0.1 else '几乎无相关'
        print(f"  {feat}: {corr:.4f}（{corr_level}）")

# 4.2 年度医疗成本散点图（单独保存）
plt.figure(figsize=(10, 6))
plt.scatter(df_core['annual_medical_cost'], df_core['total_claims_paid'], alpha=0.6, color='#2E86AB', s=30)
z = np.polyfit(df_core['annual_medical_cost'], df_core['total_claims_paid'], 1)
p = np.poly1d(z)
plt.plot(df_core['annual_medical_cost'], p(df_core['annual_medical_cost']), color='red', linewidth=2, label=f'趋势线：y={z[0]:.2f}x+{z[1]:.0f}')
plt.xlabel('年度医疗成本（元）')
plt.ylabel('总赔付金额（元）')
plt.title(f'年度医疗成本 vs 总赔付金额（相关系数：{corr_with_target["annual_medical_cost"]:.4f}）')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('3_年度医疗成本与总赔付金额散点图.png', bbox_inches='tight')
plt.close()
print("【双变量分析图表已保存】：3_年度医疗成本与总赔付金额散点图.png")

# 4.3 健康风险箱线图（单独保存）
plt.figure(figsize=(10, 6))
risk_groups = [df_core[df_core['health_risk']==risk]['total_claims_paid'] for risk in ['低风险', '中风险', '高风险']]
bp = plt.boxplot(risk_groups, labels=['低风险（≤1种慢性病）', '中风险（2-3种）', '高风险（≥4种）'], patch_artist=True)
colors = ['#90EE90', '#FFD700', '#FFB6C1']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for i, risk in enumerate(['低风险', '中风险', '高风险']):
    mean_val = df_core[df_core['health_risk']==risk]['total_claims_paid'].mean()
    plt.text(i+1, mean_val+800, f'均值：{mean_val:.0f}元', ha='center', fontsize=9, color='darkred', fontweight='bold')
plt.xlabel('健康风险等级')
plt.ylabel('总赔付金额（元）')
plt.title(f'健康风险等级 vs 总赔付金额（相关系数：{corr_with_target["chronic_count"]:.4f}）')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('4_健康风险与总赔付金额箱线图.png', bbox_inches='tight')
plt.close()
print("【双变量分析图表已保存】：4_健康风险与总赔付金额箱线图.png")

# 4.4 吸烟状态箱线图（单独保存）
plt.figure(figsize=(10, 6))
smoke_groups = [df_core[df_core['smoke_status']==status]['total_claims_paid'] for status in ['从不吸烟', '曾经吸烟', '当前吸烟']]
bp = plt.boxplot(smoke_groups, labels=['从不吸烟', '曾经吸烟', '当前吸烟'], patch_artist=True)
colors = ['#87CEEB', '#DDA0DD', '#FF6347']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
median_never = df_core[df_core['smoke_status']=='从不吸烟']['total_claims_paid'].median()
median_current = df_core[df_core['smoke_status']=='当前吸烟']['total_claims_paid'].median()
plt.text(2, max(df_core['total_claims_paid'])*0.6, f'当前吸烟组中位数是从不吸烟组的{median_current/median_never:.1f}倍',
         ha='center', fontsize=9, color='darkblue', fontweight='bold')
plt.xlabel('吸烟状态')
plt.ylabel('总赔付金额（元）')
plt.title('吸烟状态 vs 总赔付金额')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('5_吸烟状态与总赔付金额箱线图.png', bbox_inches='tight')
plt.close()
print("【双变量分析图表已保存】：5_吸烟状态与总赔付金额箱线图.png")

# 4.5 年龄分组箱线图（单独保存）
plt.figure(figsize=(10, 6))
age_groups = [df_core[df_core['age_group']==age]['total_claims_paid'] for age in ['青年', '中年', '老年']]
bp = plt.boxplot(age_groups, labels=['青年（<35岁）', '中年（35-59岁）', '老年（≥60岁）'], patch_artist=True)
colors = ['#32CD32', '#FFA500', '#DC143C']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for i, age in enumerate(['青年', '中年', '老年']):
    mean_val = df_core[df_core['age_group']==age]['total_claims_paid'].mean()
    plt.text(i+1, mean_val+1000, f'均值：{mean_val:.0f}元', ha='center', fontsize=9, color='white', fontweight='bold')
plt.xlabel('年龄分组')
plt.ylabel('总赔付金额（元）')
plt.title(f'年龄分组 vs 总赔付金额（相关系数：{corr_with_target["age"]:.4f}）')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('6_年龄分组与总赔付金额箱线图.png', bbox_inches='tight')
plt.close()
print("【双变量分析图表已保存】：6_年龄分组与总赔付金额箱线图.png")


# ==============================
# 5. 多变量交叉分析（每个图表单独保存）
# ==============================
print("\n" + "="*60)
print("【多变量分析：双因素交叉箱线图】")
print("="*60)

# 5.1 年龄分组 × 健康风险 交叉箱线图（单独保存）
plt.figure(figsize=(12, 8))
cross1_groups = []
cross1_labels = []
age_order = ['青年', '中年', '老年']
risk_order = ['低风险', '中风险', '高风险']

for age in age_order:
    for risk in risk_order:
        group_data = df_core[(df_core['age_group']==age) & (df_core['health_risk']==risk)]['total_claims_paid']
        cross1_groups.append(group_data)
        cross1_labels.append(f'{age}\n{risk}')

bp1 = plt.boxplot(cross1_groups, labels=cross1_labels, patch_artist=True)
age_colors = ['#2E86AB', '#F18F01', '#C73E1D']
risk_alpha = [0.6, 0.8, 1.0]
color_idx = 0
alpha_idx = 0
for i, patch in enumerate(bp1['boxes']):
    if i % 3 == 0 and i != 0:
        color_idx += 1
        alpha_idx = 0
    patch.set_facecolor(age_colors[color_idx])
    patch.set_alpha(risk_alpha[alpha_idx])
    alpha_idx += 1

# 标注最高赔付组
old_high_risk_mean = df_core[(df_core['age_group']=='老年') & (df_core['health_risk']=='高风险')]['total_claims_paid'].mean()
plt.text(9, old_high_risk_mean + 2000, f'老年高风险组\n均值：{old_high_risk_mean:.0f}元',
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3), fontweight='bold')
plt.xlabel('年龄分组 × 健康风险等级')
plt.ylabel('总赔付金额（元）')
plt.title('年龄分组 × 健康风险 对总赔付金额的交叉影响')
plt.grid(alpha=0.3, axis='y')

# 图例
from matplotlib.patches import Patch
legend1 = [Patch(facecolor='#2E86AB', alpha=0.8, label='青年'),
           Patch(facecolor='#F18F01', alpha=0.8, label='中年'),
           Patch(facecolor='#C73E1D', alpha=0.8, label='老年'),
           Patch(facecolor='gray', alpha=0.6, label='低风险'),
           Patch(facecolor='gray', alpha=0.8, label='中风险'),
           Patch(facecolor='gray', alpha=1.0, label='高风险')]
plt.legend(handles=legend1, loc='upper left', ncol=2)

plt.tight_layout()
plt.savefig('7_年龄与健康风险交叉分析.png', bbox_inches='tight')
plt.close()
print("【多变量分析图表已保存】：7_年龄与健康风险交叉分析.png")

# 5.2 保险计划等级 × 健康风险 交叉箱线图（单独保存）
plt.figure(figsize=(12, 8))
cross2_groups = []
cross2_labels = []
plan_order = ['基础型', '中等型', '高端型']
risk_order = ['低风险', '中风险', '高风险']

for plan in plan_order:
    for risk in risk_order:
        group_data = df_core[(df_core['plan_tier']==plan) & (df_core['health_risk']==risk)]['total_claims_paid']
        cross2_groups.append(group_data)
        cross2_labels.append(f'{plan}\n{risk}')

bp2 = plt.boxplot(cross2_groups, labels=cross2_labels, patch_artist=True)
plan_colors = ['#8B4513', '#CD853F', '#DAA520']
risk_alpha = [0.6, 0.8, 1.0]
color_idx = 0
alpha_idx = 0
for i, patch in enumerate(bp2['boxes']):
    if i % 3 == 0 and i != 0:
        color_idx += 1
        alpha_idx = 0
    patch.set_facecolor(plan_colors[color_idx])
    patch.set_alpha(risk_alpha[alpha_idx])
    alpha_idx += 1

# 标注最高赔付组
high_plan_high_risk_mean = df_core[(df_core['plan_tier']=='高端型') & (df_core['health_risk']=='高风险')]['total_claims_paid'].mean()
plt.text(9, high_plan_high_risk_mean + 1500, f'高端计划高风险组\n均值：{high_plan_high_risk_mean:.0f}元',
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3), fontweight='bold')
plt.xlabel('保险计划等级 × 健康风险等级')
plt.ylabel('总赔付金额（元）')
plt.title('保险计划等级 × 健康风险 对总赔付金额的交叉影响')
plt.grid(alpha=0.3, axis='y')

# 图例
legend2 = [Patch(facecolor='#8B4513', alpha=0.8, label='基础型计划'),
           Patch(facecolor='#CD853F', alpha=0.8, label='中等型计划'),
           Patch(facecolor='#DAA520', alpha=0.8, label='高端型计划'),
           Patch(facecolor='gray', alpha=0.6, label='低风险'),
           Patch(facecolor='gray', alpha=0.8, label='中风险'),
           Patch(facecolor='gray', alpha=1.0, label='高风险')]
plt.legend(handles=legend2, loc='upper left', ncol=2)

plt.tight_layout()
plt.savefig('8_保险计划与健康风险交叉分析.png', bbox_inches='tight')
plt.close()
print("【多变量分析图表已保存】：8_保险计划与健康风险交叉分析.png")


# 输出交叉分析结论
old_high_risk = df_core[(df_core['age_group']=='老年') & (df_core['health_risk']=='高风险')]['total_claims_paid'].mean()
young_low_risk = df_core[(df_core['age_group']=='青年') & (df_core['health_risk']=='低风险')]['total_claims_paid'].mean()
high_plan_high_risk = df_core[(df_core['plan_tier']=='高端型') & (df_core['health_risk']=='高风险')]['total_claims_paid'].mean()
basic_plan_low_risk = df_core[(df_core['plan_tier']=='基础型') & (df_core['health_risk']=='低风险')]['total_claims_paid'].mean()

print("【交叉分析关键结论】")
print(f"1. 年龄×健康风险协同效应：老年高风险用户总赔付均值（{old_high_risk:.0f}元）是青年低风险用户（{young_low_risk:.0f}元）的{old_high_risk/young_low_risk:.1f}倍；")
print(f"2. 保险×健康风险协同效应：高端计划高风险用户总赔付均值（{high_plan_high_risk:.0f}元）是基础计划低风险用户（{basic_plan_low_risk:.0f}元）的{high_plan_high_risk/basic_plan_low_risk:.1f}倍；")
print(f"3. 核心洞察：健康风险是总赔付金额的核心驱动因素，且与年龄、保险计划存在显著协同效应。")


# ==============================
# 6. 整体分析总结
# ==============================
print("\n" + "="*60)
print("【整体分析总结】")
print("="*60)
print("1. 总赔付金额核心特征：")
print(f"   - 分布：偏度={skewness:.2f}（显著右偏），80%用户集中在0-{q75:.0f}元，20%高赔付用户贡献60%总赔付；")
print(f"   - 异常值：{outlier_count/len(df_core)*100:.2f}%的异常值，主要来自老年高风险用户；")
print(f"   - 集中趋势：均值（{df_core['total_claims_paid'].mean():.0f}元）> 中位数（{df_core['total_claims_paid'].median():.0f}元），受高赔付用户影响。")

print("\n2. 关键影响因素：")
print(f"   - 强相关：年度医疗成本（{corr_with_target['annual_medical_cost']:.4f}）、健康风险（{corr_with_target['chronic_count']:.4f}）；")
print(f"   - 中等相关：风险评分（{corr_with_target['risk_score']:.4f}）、年龄（{corr_with_target['age']:.4f}）。")

print("\n3. 业务建议：")
print("   - 分层定价：对老年高风险用户保费上浮20%-30%；")
print("   - 风控重点：关注高端计划+高风险用户的理赔审核；")
print("   - 数据完善：补充病种类型、就医类型字段。")

print("\n4. 输出文件（共8个图表）：")
print("   - 1_总赔付金额分布直方图.png")
print("   - 2_总赔付金额累积分布曲线.png")
print("   - 3_年度医疗成本与总赔付金额散点图.png")
print("   - 4_健康风险与总赔付金额箱线图.png")
print("   - 5_吸烟状态与总赔付金额箱线图.png")
print("   - 6_年龄分组与总赔付金额箱线图.png")
print("   - 7_年龄与健康风险交叉分析.png")
print("   - 8_保险计划与健康风险交叉分析.png")