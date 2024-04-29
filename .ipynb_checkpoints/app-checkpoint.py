import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile

# Define a function to load data
@st.cache  # Use caching to load the data only once
def load_data():
    file_path = "inputs/practice_file.csv"
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        return data
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
#def load_data():
    # Define the path to the .dta file
    #ccm_youngfirms = pd.read_csv("inputs/practice_file.csv")

    # # Check if the .dta file exists, if not, unzip it
    # if not os.path.exists(ccm_youngfirms):
    #     zip_path = "inputs/ccm_youngfirms_2000_2018.zip"
    #     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #         zip_ref.extractall("inputs/ccm_youngfirms_2000_2018.dta")

    # # Load the data from the .dta file
    # data = pd.read_stata(ccm_youngfirms)
    # return data
data = load_data()

# Dashboard setup
st.title('SPAC and IPO Dashboard')
st.write('This dashboard visualizes data for young firms from the CCM dataset.')

# Use columns for layout
col1, col2 = st.columns((1, 2))

# Inputs on the left column
with col1:
    st.subheader("User Inputs")
    year = st.number_input('Year', min_value=2000, max_value=2018, value=2018)
    industry_sector = st.selectbox('Industry Sector', options=data['Industry'].unique())
    market_cap = st.number_input('Market Capitalization', min_value=0, step=1, format='%d')
    profitability_metrics = st.multiselect('Profitability Metrics', options=['ROA', 'ROE'])
    funding_needs = st.number_input('Funding Needs', min_value=0, step=1, format='%d')
    revenue_growth_rate = st.slider('Revenue Growth Rate', min_value=-100.0, max_value=100.0, value=0.0)

# Plots on the right column
with col2:
    st.subheader("Data Visualizations")
    if st.button('Show Plots'):
        fig, ax = plt.subplots()
        data['mkvalt'].plot(kind='hist')
        plt.title('Histogram of Market Cap')
        st.pyplot(fig)


