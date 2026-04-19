import streamlit as st
import pandas as pd
import numpy as np
import scikit-learn as sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# 🔹 PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Factory Optimization System",
    layout="wide",
    page_icon="📦"
)

# =========================================================
# 🔹 LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Nassau Candy Distributor.csv", encoding='latin1')
    df.columns = df.columns.str.strip()

    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

    df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days
    return df

df = load_data()

# =========================================================
# 🔹 MODEL TRAINING
# =========================================================
features = ['Product Name', 'City', 'Region', 'Ship Mode']
X = df[features].copy()
y = df['Lead Time']

le_dict = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

model = RandomForestRegressor()
model.fit(X, y)

# =========================================================
# 🔹 HEADER
# =========================================================
st.title("📦 Factory Reallocation & Optimization System")
st.markdown("Optimize product allocation across factories using ML-driven insights")

# =========================================================
# 🔹 SIDEBAR
# =========================================================
st.sidebar.header("🔧 Controls")

product = st.sidebar.selectbox("Select Product", df['Product Name'].unique())
region = st.sidebar.selectbox("Select Region", df['Region'].unique())
ship_mode = st.sidebar.selectbox("Ship Mode", df['Ship Mode'].unique())

priority = st.sidebar.slider("Optimization Priority (Speed ↔ Profit)", 0, 100, 50)

# =========================================================
# 🔹 SIMULATION FUNCTION
# =========================================================
def simulate(product, region, ship_mode):
    factories = df['City'].unique()
    results = []

    for f in factories:
        temp = pd.DataFrame({
            'Product Name': [product],
            'City': [f],
            'Region': [region],
            'Ship Mode': [ship_mode]
        })

        for col in temp.columns:
            le = le_dict[col]
            temp[col] = le.transform(temp[col].astype(str))

        pred = model.predict(temp)[0]

        results.append({
            "Factory": f,
            "Predicted Lead Time": round(pred,2)
        })

    result_df = pd.DataFrame(results)

    base = result_df['Predicted Lead Time'].mean()
    result_df['Improvement'] = base - result_df['Predicted Lead Time']
    result_df['Profit Impact (%)'] = result_df['Improvement'] * 2

    return result_df.sort_values(by="Predicted Lead Time")

result = simulate(product, region, ship_mode)

# =========================================================
# 🔹 KPI SECTION
# =========================================================
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Best Lead Time", round(result['Predicted Lead Time'].min(),2))
col2.metric("Avg Lead Time", round(result['Predicted Lead Time'].mean(),2))
col3.metric("Max Improvement", round(result['Improvement'].max(),2))
col4.metric("Avg Profit Impact", round(result['Profit Impact (%)'].mean(),2))

# =========================================================
# 🔹 TABS (PRO UI)
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏭 Simulator",
    "🔍 What-If Analysis",
    "⭐ Recommendations",
    "⚠️ Risk Panel"
])

# =========================================================
# 🔹 TAB 1: SIMULATOR
# =========================================================
with tab1:
    st.subheader("Factory Performance Comparison")
    st.dataframe(result, use_container_width=True)

# =========================================================
# 🔹 TAB 2: WHAT-IF
# =========================================================
with tab2:
    st.subheader("Lead Time Comparison")

    st.bar_chart(result.set_index("Factory")['Predicted Lead Time'])

    st.subheader("Improvement Distribution")
    st.line_chart(result.set_index("Factory")['Improvement'])

# =========================================================
# 🔹 TAB 3: RECOMMENDATIONS
# =========================================================
with tab3:
    st.subheader("Top Factory Recommendations")

    if priority > 50:
        sorted_df = result.sort_values(
            by=["Predicted Lead Time", "Profit Impact (%)"],
            ascending=[True, False]
        )
    else:
        sorted_df = result.sort_values(
            by=["Profit Impact (%)", "Predicted Lead Time"],
            ascending=[False, True]
        )

    top = sorted_df.head(5)

    st.success("✅ Recommended Factory Options")
    st.dataframe(top, use_container_width=True)

# =========================================================
# 🔹 TAB 4: RISK PANEL
# =========================================================
with tab4:
    st.subheader("High Risk Assignments")

    risk = result[result['Improvement'] < 0]

    if risk.empty:
        st.success("No high-risk assignments detected 🎉")
    else:
        st.warning("⚠️ Risky factory allocations")
        st.dataframe(risk)

    st.subheader("Profit Impact Visualization")
    st.bar_chart(result.set_index("Factory")['Profit Impact (%)'])

# =========================================================
# 🔹 FOOTER
# =========================================================
st.markdown("---")
st.caption("Built with Machine Learning for Smart Supply Chain Optimization 🚀")

