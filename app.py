import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile

# Define a function to load data
@st.cache  # Use caching to load the data only once
def load_data():
    file_path = "inputs/master_filtered_data.csv"
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        return data
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
data = load_data()

# Dashboard setup
st.title('FIN 377-Final Project-Team 12: SPAC and IPO Dashboard')
st.write('This dashboard visualizes data for young firms from the CCM dataset.')


# Use columns for layout
col1, col2 = st.columns((1, 2))

# Inputs on the left column
with col1:
    st.subheader("User Inputs")
    year = st.number_input('Year', min_value=2000, max_value=2018, value=2018)
    #industry_sector = st.selectbox('Industry Sector', options=data['Industry'].unique())
    market_cap = st.number_input('Market Capitalization', min_value=0, step=1, format='%d')
    profitability_metrics = st.multiselect('Profitability Metrics', options=['ROA', 'ROE'])
    funding_needs = st.number_input('Funding Needs', min_value=0, step=1, format='%d')
    revenue_growth_rate = st.slider('Revenue Growth Rate', min_value=-100.0, max_value=100.0, value=0.0)

# Plots on the right column
with col2:
    st.subheader("Data Visualizations")
    if st.button('Show Plots'):
        fig, ax = plt.subplots()
        data['mkvalt'].plot(kind='line')
        plt.title('Histogram of Market Cap')
        st.pyplot(fig)

