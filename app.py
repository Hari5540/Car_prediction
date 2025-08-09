import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="🚗 Car Price Predictor", layout="wide")

# ------------------- THEME TOGGLE -------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

selected_theme = st.sidebar.radio("🌗 Select Theme", ["light", "dark"], index=0 if st.session_state.theme == "light" else 1)
st.session_state.theme = selected_theme

# Inject CSS based on theme
if st.session_state.theme == "light":
    theme_css = """
    <style>
    :root {
      --bg: #ffffff;
      --text: #0b0b0b;
      --sidebar: #f8f9fb;
      --card: #ffffff;
    }
    </style>
    """
else:
    theme_css = """
    <style>
    :root {
      --bg: #0e1116;
      --text: #e6e6e6;
      --sidebar: #111315;
      --card: #1a1a1a;
    }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)

# Apply styles globally
st.markdown("""
<style>
[data-testid="stAppViewContainer"], .block-container {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}
[data-testid="stSidebar"] {
  background-color: var(--sidebar) !important;
  color: var(--text) !important;
}
.stButton>button {
  background-color: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
}
input, textarea, select {
  color: var(--text) !important;
  background-color: var(--card) !important;
}
[data-testid="stMarkdownContainer"], [data-testid="stDataFrame"] {
  color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------- DATA LOADING -------------------
df = pd.read_csv("Cleaned_Car_data.csv")
df = df.dropna()
df = df[df['Price'].str.replace(',', '').str.replace(' Ask For Price', '').str.isnumeric()]
df['Price'] = df['Price'].str.replace(',', '').str.replace(' Ask For Price', '').astype(int)
df['kms_driven'] = df['kms_driven'].str.replace(' kms', '').str.replace(',', '')
df = df[df['kms_driven'].str.isnumeric()]
df['kms_driven'] = df['kms_driven'].astype(int)
df = df[df['year'].str.isnumeric()]
df['year'] = df['year'].astype(int)
df = df[df['fuel_type'].notna()]

# Extract brand/company
car_names = df['name'].str.split(' ', expand=True)
df['brand'] = car_names[0]
df['company'] = car_names[1]
df = df.dropna(subset=['brand', 'company', 'fuel_type'])

# ------------------- SIDEBAR FILTERS -------------------
st.sidebar.header("🔧 Filter Car Details")
selected_company = st.sidebar.selectbox("Select Model", sorted(df['company'].unique()))
brands_available = df[df['company'] == selected_company]['brand'].unique()
selected_brand = st.sidebar.selectbox("Select Brand", sorted(brands_available))
selected_fuel = st.sidebar.selectbox("Select Fuel Type", sorted(df['fuel_type'].unique()))
selected_kms = st.sidebar.slider("KMs Driven", int(df['kms_driven'].min()), int(df['kms_driven'].max()), 55000, step=1000)
selected_year = st.sidebar.slider("Year of Purchase", int(df['year'].min()), int(df['year'].max()), 1998)

# Filter display data
df_filtered_display = df[(df['company'] == selected_company) &
                         (df['brand'] == selected_brand) &
                         (df['fuel_type'] == selected_fuel)]

# ------------------- PRICE PREDICTION -------------------
if st.sidebar.button("🔮 Predict Price"):
    X = df[['kms_driven', 'year']]
    y = df['Price']
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict([[selected_kms, selected_year]])[0]
    prediction = max(0, int(prediction))
    st.sidebar.success(f"Predicted Price: ₹ {prediction:,}")

# ------------------- MAIN CONTENT -------------------
st.markdown("<h1 style='text-align: center;'>📊 Car Price Insights</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    ax1.hist(df['Price'], bins=40, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Price (₹)')
    ax1.set_ylabel('Number of Cars')
    ax1.set_title('Car Price Distribution')
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['kms_driven'], df['Price'], alpha=0.5)
    ax2.set_xlabel('KMs Driven')
    ax2.set_ylabel('Price (₹)')
    ax2.set_title('Price vs KMs Driven')
    ax2.ticklabel_format(style='plain', axis='y')
    st.pyplot(fig2)

if not df_filtered_display.empty:
    st.markdown("### 🧪 Preview Filtered Car Data")
    st.dataframe(df_filtered_display[['name', 'company', 'brand', 'year', 'kms_driven', 'fuel_type', 'Price']].head())
else:
    st.warning("⚠️ No data found for the selected filter combination.")

