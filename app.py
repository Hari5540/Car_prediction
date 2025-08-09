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

