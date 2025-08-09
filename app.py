import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    st.title("Car Price Predictor")
    # rest of your app

if __name__ == "__main__":
    main()

st.set_page_config(page_title="🚗 Car Price Predictor", layout="wide")

# --- Theme button with Light/Dark options (drop-in; paste right after st.set_page_config) ---
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "show_theme" not in st.session_state:
    st.session_state.show_theme = False

# toggle show/hide of theme options
if st.button("🎨 Theme"):
    st.session_state.show_theme = not st.session_state.show_theme

# show radio options only when requested
if st.session_state.show_theme:
    choice = st.radio("Choose theme", ["Light", "Dark"],
                      index=0 if st.session_state.theme == "light" else 1,
                      key="theme_radio")
    # apply immediately
    st.session_state.theme = choice.lower()
    st.experimental_rerun()  # ensures CSS/JS update applies immediately

# CSS that uses a data-theme attribute (targets Streamlit containers)
_theme_css = """
<style>
:root {
  --bg: #ffffff;
  --text: #0b0b0b;
  --sidebar: #f8f9fb;
  --card: #ffffff;
}

/* dark mode vars */
[data-theme="dark"] {
  --bg: #0e1116;
  --text: #e6e6e6;
  --sidebar: #111315;
  --card: #1a1a1a;
}

/* main app container */
[data-testid="stAppViewContainer"], .block-container {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* sidebar */
[data-testid="stSidebar"] {
  background-color: var(--sidebar) !important;
  color: var(--text) !important;
}

/* headers / markdown */
[data-testid="stMarkdownContainer"] {
  color: var(--text) !important;
}

/* buttons / cards */
.stButton>button, .css-1d391kg, .css-1b08t9f {
  background-color: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}

/* tables / dataframe container fallback */
[data-testid="stDataFrame"] {
  color: var(--text) !important;
  background-color: transparent !important;
}

/* small fixes for inputs & labels */
input, textarea, select {
  color: var(--text) !important;
  background-color: var(--card) !important;
}
</style>
"""
st.markdown(_theme_css, unsafe_allow_html=True)

# set data-theme attribute on <html> so CSS variables apply
if st.session_state.theme == "dark":
    st.markdown("<script>document.documentElement.setAttribute('data-theme', 'dark');</script>", unsafe_allow_html=True)
else:
    st.markdown("<script>document.documentElement.setAttribute('data-theme', 'light');</script>", unsafe_allow_html=True)

# -------------------------
# Optional (recommended) helper to make matplotlib plots follow the theme.
# Use this inside your plotting code BEFORE showing the plot:
#
# bg = "#0e1116" if st.session_state.theme == "dark" else "#ffffff"
# fg = "#e6e6e6" if st.session_state.theme == "dark" else "#000000"
# fig.patch.set_facecolor(bg)
# ax.set_facecolor(bg)
# ax.title.set_color(fg)
# ax.xaxis.label.set_color(fg)
# ax.yaxis.label.set_color(fg)
# ax.tick_params(colors=fg)
# for spine in ax.spines.values():
#     spine.set_color(fg)
#
# Example: after fig, ax = plt.subplots()
# then run the block above (adapt names) and then st.pyplot(fig)
# -------------------------

# Load and clean data
df = pd.read_csv("Cleaned_Car_data.csv")

df = df.dropna()
df = df[df['Price'].str.replace(',', '').str.replace(' Ask For Price', '').str.isnumeric()]
df['Price'] = df['Price'].str.replace(',', '').str.replace(' Ask For Price', '')
df['Price'] = df['Price'].astype(int)

df['kms_driven'] = df['kms_driven'].str.replace(' kms', '').str.replace(',', '')
df = df[df['kms_driven'].str.isnumeric()]
df['kms_driven'] = df['kms_driven'].astype(int)

df = df[df['year'].str.isnumeric()]
df['year'] = df['year'].astype(int)

df = df[df['fuel_type'].notna()]

# Extract brand and company
car_names = df['name'].str.split(' ', expand=True)
df['brand'] = car_names[0]
df['company'] = car_names[1]

df = df.dropna(subset=['brand', 'company', 'fuel_type'])

# Sidebar UI
st.sidebar.header("🔧 Filter Car Details")
selected_company = st.sidebar.selectbox("Select Model", sorted(df['company'].unique()))
brands_available = df[df['company'] == selected_company]['brand'].unique()
selected_brand = st.sidebar.selectbox("Select Brand", sorted(brands_available))
selected_fuel = st.sidebar.selectbox("Select Fuel Type", sorted(df['fuel_type'].unique()))
selected_kms = st.sidebar.slider("KMs Driven", int(df['kms_driven'].min()), int(df['kms_driven'].max()), 55000, step=1000)
selected_year = st.sidebar.slider("Year of Purchase", int(df['year'].min()), int(df['year'].max()), 1998)

# Filter display data for charts and preview
df_filtered_display = df[(df['company'] == selected_company) &
                         (df['brand'] == selected_brand) &
                         (df['fuel_type'] == selected_fuel)]

# Predict Price
if st.sidebar.button("🔮 Predict Price"):
    X = df[['kms_driven', 'year']]
    y = df['Price']
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict([[selected_kms, selected_year]])[0]
    prediction = max(0, int(prediction))  # Prevent negatives
    st.sidebar.success(f"Predicted Price: ₹ {prediction:,}")

# Main Dashboard
st.markdown("""
    <h1 style='text-align: center;'>📊 Car Price Insights</h1>
    <h3 style='text-align: center;'>Distribution of Car Prices</h3>
""", unsafe_allow_html=True)

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

# Display Filtered Data
if not df_filtered_display.empty:
    st.markdown("### 🧪 Preview Filtered Car Data")
    st.dataframe(df_filtered_display[['name', 'company', 'brand', 'year', 'kms_driven', 'fuel_type', 'Price']].head())
else:
    st.warning("⚠️ No data found for the selected filter combination.")





