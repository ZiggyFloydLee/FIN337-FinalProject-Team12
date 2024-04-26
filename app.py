import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile

# Define a function to load data
@st.cache  # Use caching to load the data only once
def load_data():
    # Define the path to the .dta file
    ccm_youngfirms = "inputs/ccm_youngfirms_2000_2018.dta/ccm_youngfirms_2000_2018.dta"

    # Check if the .dta file exists, if not, unzip it
    if not os.path.exists(ccm_youngfirms):
        zip_path = "inputs/ccm_youngfirms_2000_2018.zip"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("inputs/ccm_youngfirms_2000_2018.dta")

    # Load the data from the .dta file
    data = pd.read_stata(ccm_youngfirms)
    return data

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
    industry_sector = st.selectbox('Industry Sector', options=data['sich'].unique())
    

# Plots on the right column
with col2:
    st.subheader("Data Visualizations")
    if st.button('Show Plots'):
        fig, ax = plt.subplots()
        data['mkvalt'].plot(kind='hist')
        plt.title('Histogram of Market Cap')
        st.pyplot(fig)


