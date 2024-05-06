
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load data
merged_df = pd.read_csv('inputs/master_filtered_data.csv', low_memory=False)
master_merge = pd.read_csv('inputs/masterMerge.csv')

# Define features and target
features = ['revt', 'cshi', 'naicsh']  # Ensure these column names exist in merged_df
target = 'IS_SPAC'  # Ensure this column exists in merged_df

# Clean data
data_clean = merged_df.dropna(subset=features + [target])

# Separate features and target
X = data_clean[features]
y = data_clean[target]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train nearest neighbors model
n_neighbors = 5
nn = NearestNeighbors(n_neighbors=n_neighbors)
nn.fit(X_scaled)

# Streamlit app setup
st.title('SPAC and IPO Classifier')
st.sidebar.subheader('User Inputs')

# User inputs
user_inputs = {}
for feature in features:
    user_inputs[feature] = st.sidebar.number_input(f'Enter {feature}', value=X[feature].mean())

# Predict and find nearest neighbors
input_data = np.array([user_inputs[feature] for feature in features]).reshape(1, -1)
input_scaled = scaler.transform(input_data)
distances, indices = nn.kneighbors(input_scaled)

# Display nearest neighbors
st.write('### Results:')
st.write('Based on your inputs, your firm is most similar to the following firms:')
for i, idx in enumerate(indices[0]):
    st.write(f"Firm {i + 1}: {master_merge.iloc[idx]['tic']}")
predicted_spac = 'SPAC' if y.iloc[indices[0]].mean() > 0.5 else 'not a SPAC'
st.write(f'Based on the characteristics of these firms, your firm is most likely {predicted_spac}')

# Plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5)
legend = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend)
input_pca = pca.transform(input_scaled)
ax.scatter(input_pca[0, 0], input_pca[0, 1], c='red', s=100, label='Your Firm')
ax.set_title('PCA Plot of Firms with Nearest Neighbors')
ax.legend()
st.pyplot(fig)


