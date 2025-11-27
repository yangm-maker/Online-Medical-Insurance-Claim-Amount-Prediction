# pages/4_数据库数据分析.py
import streamlit as st
import pandas as pd
from utils import set_chinese_font, load_all_resources
from utils import MYSQL_CONFIG, create_engine

# ---------------------- 初始化配置 ----------------------
set_chinese_font()
resources, load_status = load_all_resources()

# 页面配置
st.set_page_config(
    page_title="数据库数据查询",
    page_icon="📋",
    layout="wide"
)

# 加载失败处理
if resources is None:
    st.error(f"❌ 资源加载失败：{load_status}")
    st.stop()


# ---------------------- 核心函数：从MySQL加载数据 ----------------------
@st.cache_data(ttl=300)  # 缓存5分钟
def load_data_from_mysql():
    try:
        conn_str = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}?charset={MYSQL_CONFIG['charset']}"
        engine = create_engine(conn_str)
        df = pd.read_sql("SELECT * FROM prediction_records", con=engine)
        st.success(f"✅ 成功加载 {len(df)} 条历史预测记录")
        return df
    except Exception as e:
        st.error(f"❌ 数据库读取失败：{str(e)}")
        return pd.DataFrame()


# ---------------------- 主内容 ----------------------
st.title("📋 数据库历史预测记录查询")
st.divider()

# 加载数据
df = load_data_from_mysql()

if not df.empty:
    # 数据预览
    st.subheader("📄 历史预测记录预览")
    st.dataframe(df, use_container_width=True)

    # 数据下载
    st.subheader("💾 数据下载")
    csv_data = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="下载全部记录为CSV",
        data=csv_data,
        file_name="历史预测记录_全部.csv",
        mime="text/csv"
    )

    # 统计信息
    st.subheader("📊 记录统计")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总记录数", f"{len(df):,} 条")
    with col2:
        st.metric("涉及模型数量", f"{df['model_name'].nunique()} 个")
    with col3:
        st.metric("最早记录时间", df["predict_time"].min().strftime("%Y-%m-%d %H:%M:%S"))
else:
    st.warning("⚠️ 暂无历史预测记录，可通过「在线预测」页面生成数据后再查询")
    st.markdown("👉 操作指引：进入「在线预测」页面 → 填写信息 → 提交预测 → 数据自动存入数据库")