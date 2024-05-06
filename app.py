# import streamlit as st
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import numpy as np

# # Load data
# merged_df = pd.read_csv('inputs/master_filtered_data.csv', low_memory=False)
# master_merge = pd.read_csv('inputs/tickers_filtered.csv')

# # Define features and target
# features = ['naicsh', 'caps', 'cshi', 'epspi', 'mkvalt']  # Ensure these column names exist in merged_df
# target = 'IS_SPAC'  # Ensure this column exists in merged_df

# # Clean data
# data_clean = merged_df.dropna(subset=features + [target])

# # Separate features and target
# X = data_clean[features]
# y = data_clean[target]

# # Feature scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train nearest neighbors model
# n_neighbors = 5
# nn = NearestNeighbors(n_neighbors=n_neighbors)
# nn.fit(X_scaled)

# # Streamlit app setup
# st.title('SPAC and IPO Classifier')
# st.sidebar.subheader('User Inputs')

# # User inputs
# user_inputs = {}
# for feature in features:
#     user_inputs[feature] = st.sidebar.number_input(f'Enter {feature}', value=X[feature].mean())

# # Predict and find nearest neighbors
# input_data = np.array([user_inputs[feature] for feature in features]).reshape(1, -1)
# input_scaled = scaler.transform(input_data)
# distances, indices = nn.kneighbors(input_scaled)

# # Display nearest neighbors
# st.write('### Results:')
# st.write('Based on your inputs, your firm is most similar to the following firms:')
# for i, idx in enumerate(indices[0]):
#     st.write(f"Firm {i + 1}: {master_merge.iloc[idx]['tic']}")
# predicted_spac = 'SPAC' if y.iloc[indices[0]].mean() > 0.5 else 'not a SPAC'
# st.write(f'Based on the characteristics of these firms, your firm is most likely {predicted_spac}')
# # st.write("Nearest Neighbors' Indicies:", indices)

# # For viewing the actual SPAC status of these neighbors
# st.write("Nearest Neighbors' SPAC Status:")
# st.write(y.iloc[indices[0]].values)

# # Plotting
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# fig, ax = plt.subplots()
# scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5)
# legend = ax.legend(*scatter.legend_elements(), title="Classes", loc='lower right')
# ax.add_artist(legend)
# input_pca = pca.transform(input_scaled)
# ax.scatter(input_pca[0, 0], input_pca[0, 1], c='red', s=100, label='Your Firm')
# ax.set_title('PCA Plot of Firms with Nearest Neighbors')
# ax.legend()
# st.pyplot(fig)

import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load data
merged_df = pd.read_csv('inputs/master_filtered_data.csv', low_memory=False)
master_merge = pd.read_csv('inputs/tickers_filtered.csv')

# Define features and target
features = ['naicsh', 'caps', 'cshi', 'epspi', 'mkvalt']
target = 'IS_SPAC'

# Clean data
data_clean = merged_df.dropna(subset=features + [target])

# Separate features and target
X = data_clean[features]
y = data_clean[target]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define clearer labels for user inputs
feature_labels = {
    'naicsh': 'NAICS Code',
    'caps': 'Capital Stock',
    'cshi': 'Common Shares Issued',
    'epspi': 'Earnings Per Share',
    'mkvalt': 'Market Value Total'
}

# Train nearest neighbors model
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)

# Streamlit app setup
st.title('SPAC and IPO Classifier')
st.sidebar.subheader('User Inputs')

# User inputs using clearer, descriptive labels
user_inputs = {}
for feature in features:
    label = feature_labels.get(feature, feature)  # Get user-friendly labels
    user_inputs[feature] = st.sidebar.number_input(f'Enter {label}', value=X[feature].mean())

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
# For viewing the actual SPAC status of these neighbors
st.write("Nearest Neighbors' SPAC Status:")
st.write(y.iloc[indices[0]].values)

# Plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5)
legend = ax.legend(*scatter.legend_elements(), title="Classes", loc='lower right')
ax.add_artist(legend)
input_pca = pca.transform(input_scaled)
ax.scatter(input_pca[0, 0], input_pca[0, 1], c='red', s=100, label='Your Firm')
ax.set_title('PCA Plot of Firms with Nearest Neighbors')
ax.legend()
st.pyplot(fig)



# import streamlit as st
# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # Load the data
# @st.cache  # Use caching to load the data only once
# def load_data():
#     merged_df = pd.read_csv('inputs/master_filtered_data.csv', low_memory=False)
#     return merged_df

# merged_df = load_data()

# # Define features and target
# features = ['adrr', 'curuscn', 'scf', 'src', 'acominc', 'acox', 'at', 'am', 'ao', 'aoloch', 'aox', 'ap', 'at.1', 'caps', 'capx', 'cb',
#             'ch', 'che', 'clg', 'cogs', 'csho', 'cshrt', 'cstk', 'dd', 'dlc', 'dn', 'do', 'dt', 'ebit', 'ebitda', 'epspi', 'fca', 'ffo',
#             'gdwl', 'gp', 'ib', 'intan', 'invt', 'lt', 'lct', 'ni', 'niadj', 'np', 'pi', 'ppegt', 'pnrsho', 'ppent', 're', 'revt',
#             'sale', 'seq', 'tdc', 'teq', 'tstk', 'txt', 'wcap', 'naicsh', 'mkvalt', 'acchg', 'accrt', 'amc', 'ano', 'arce', 'cshi',
#             'depc', 'derhedgl']
# target = 'IS_SPAC'

# # Streamlit sidebar user inputs for prediction
# user_data = {}
# st.sidebar.header("User Input Features")
# for feature in features:
#     # Use median as a placeholder value for inputs
#     user_data[feature] = st.sidebar.number_input(f"Input {feature}", value=merged_df[feature].median())

# # Preparing the data
# valid_features = [col for col in features if col in merged_df.columns]
# data = merged_df[valid_features + [target]].dropna()

# # Feature Engineering
# imputer = SimpleImputer(strategy='mean')
# scaler = StandardScaler()

# X = imputer.fit_transform(data[valid_features])
# X = scaler.fit_transform(X)
# y = data[target].values

# # Splitting the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train KNN Classifier
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)

# # Prediction
# user_input_transformed = scaler.transform([list(user_data.values())])
# prediction = knn.predict(user_input_transformed)
# prediction_probability = knn.predict_proba(user_input_transformed)[0]

# # Display prediction results
# st.write(f"Prediction: {'SPAC' if prediction[0] else 'Non-SPAC'}")
# st.write(f"Prediction Probability: {prediction_probability.max() * 100:.2f}%")

# # Metrics display
# st.write("### Evaluation Metrics:")
# y_pred = knn.predict(X_test)
# st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
# st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
# st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

# # Plotting PCA and decision boundaries
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# knn_pca = KNeighborsClassifier(n_neighbors=5)
# knn_pca.fit(X_pca, y)

# # Create meshgrid for decision boundaries
# x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
# y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
# xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# # Predict classifications over the grid
# Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# # Plot
# fig, ax = plt.subplots(figsize=(10, 6))
# contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
# scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50)
# legend = ax.legend(*scatter.legend_elements(), title="Classes")
# ax.add_artist(legend)
# ax.set_title('K-Nearest Neighbors Classification with PCA')
# st.pyplot(fig)


# import streamlit as st
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import numpy as np

# # Load data
# merged_df = pd.read_csv('inputs/master_filtered_data.csv', low_memory=False)
# master_merge = pd.read_csv('inputs/masterMerge.csv')

# # Define all potential features (for example)
# all_potential_features = list(merged_df.columns.difference(['IS_SPAC']))  # Assuming 'IS_SPAC' is your only non-feature column
# target = 'IS_SPAC'


# # Optionally filter features based on data availability or relevance
# features = [col for col in all_potential_features if merged_df[col].notna().sum() > merged_df.shape[0] * 0.8]  # only use columns with less than 20% missing values

# # Clean data
# data_clean = merged_df.dropna(subset=features + [target])

# # Separate features and target
# X = data_clean[features]
# y = data_clean[target]

# # Feature scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train nearest neighbors model
# n_neighbors = 5
# nn = NearestNeighbors(n_neighbors=n_neighbors)
# nn.fit(X_scaled)

# # Streamlit app setup
# st.title('SPAC and IPO Classifier')
# st.sidebar.subheader('User Inputs')

# # User inputs
# user_inputs = {}
# for feature in features:
#     user_inputs[feature] = st.sidebar.number_input(f'Enter {feature}', value=X[feature].mean())

# # Predict and find nearest neighbors
# input_data = np.array([user_inputs[f] for f in features]).reshape(1, -1)
# input_scaled = scaler.transform(input_data)
# distances, indices = nn.kneighbors(input_scaled)

# # Display nearest neighbors
# st.write('### Results:')
# st.write('Based on your inputs, your firm is most similar to the following firms:')
# for i, idx in enumerate(indices[0]):
#     st.write(f"Firm {i + 1}: {master_merge.iloc[idx]['tic']}")
# predicted_spac = 'SPAC' if y.iloc[indices[0]].mean() > 0.5 else 'not a SPAC'
# st.write(f'Based on the characteristics of these firms, your firm is most likely {predicted_spac}')

# # Plotting
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# fig, ax = plt.subplots()
# scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5)
# legend = ax.legend(*scatter.legend_elements(), title="Classes")
# ax.add_artist(legend)
# input_pca = pca.transform(input_scaled)
# ax.scatter(input_pca[0, 0], input_pca[0, 1], c='red', s=100, label='Your Firm')
# ax.set_title('PCA Plot of Firms with Nearest Neighbors')
# ax.legend()
# st.pyplot(fig)


# import streamlit as st
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the data
# @st.cache
# def load_data():
#     data = pd.read_csv('inputs/master_filtered_data.csv', low_memory=False)
#     master_merge = pd.read_csv('inputs/masterMerge.csv')
#     return data, master_merge

# data, master_merge = load_data()

# # Streamlit app setup
# st.title('SPAC and IPO Classifier')
# tab1, tab2 = st.tabs(["Nearest Neighbors Model", "Classification Model"])

# # Define target
# target = 'IS_SPAC'

# # Tab 1: Nearest Neighbors Model
# with tab1:
#     st.subheader("Nearest Neighbors Analysis")
    
#     # Define features for nearest neighbors
#     features_nn = ['revt', 'cshi', 'naicsh']
    
#     # Prepare data for nearest neighbors
#     data_nn = data.dropna(subset=features_nn + [target])
#     X_nn = data_nn[features_nn]
#     y_nn = data_nn[target]

#     # Feature scaling
#     scaler_nn = StandardScaler()
#     X_scaled_nn = scaler_nn.fit_transform(X_nn)

#     # Train nearest neighbors model
#     nn = NearestNeighbors(n_neighbors=5)
#     nn.fit(X_scaled_nn)

#     # User inputs
#     user_inputs_nn = {feature: st.number_input(f'Enter {feature}', value=X_nn[feature].mean()) for feature in features_nn}
#     input_data_nn = np.array([user_inputs_nn[feature] for feature in features_nn]).reshape(1, -1)
#     input_scaled_nn = scaler_nn.transform(input_data_nn)
#     distances, indices = nn.kneighbors(input_scaled_nn)

#     # Display nearest neighbors
#     st.write('### Results:')
#     st.write('Based on your inputs, your firm is most similar to the following firms:')
#     for i, idx in enumerate(indices[0]):
#         st.write(f"Firm {i + 1}: {master_merge.iloc[idx]['tic']}")
#     predicted_spac = 'SPAC' if y_nn.iloc[indices[0]].mean() > 0.5 else 'not a SPAC'
#     st.write(f'Based on the characteristics of these firms, your firm is most likely {predicted_spac}')

#     # Plotting
#     pca_nn = PCA(n_components=2)
#     X_pca_nn = pca_nn.fit_transform(X_scaled_nn)
#     fig_nn, ax_nn = plt.subplots()
#     scatter_nn = ax_nn.scatter(X_pca_nn[:, 0], X_pca_nn[:, 1], c=y_nn, cmap='coolwarm', alpha=0.5)
#     legend_nn = ax_nn.legend(*scatter_nn.legend_elements(), title="Classes")
#     ax_nn.add_artist(legend_nn)
#     input_pca_nn = pca_nn.transform(input_scaled_nn)
#     ax_nn.scatter(input_pca_nn[0, 0], input_pca_nn[0, 1], c='red', s=100, label='Your Firm')
#     ax_nn.set_title('PCA Plot of Firms with Nearest Neighbors')
#     ax_nn.legend()
#     st.pyplot(fig_nn)

# # Tab 2: Classification Model
# with tab2:
#     # Load data
#     merged_df = pd.read_csv('inputs/master_filtered_data.csv', low_memory=False)
#     master_merge = pd.read_csv('inputs/masterMerge.csv')

#     # Define all potential features (for example)
#     all_potential_features = list(merged_df.columns.difference(['IS_SPAC']))  # Assuming 'IS_SPAC' is your only non-feature column
#     target = 'IS_SPAC'

#     # Optionally filter features based on data availability or relevance
#     features = [col for col in all_potential_features if merged_df[col].notna().sum() > merged_df.shape[0] * 0.8]  # only use columns with less than 20% missing values

#     # Clean data
#     data_clean = merged_df.dropna(subset=features + [target])

#     # Separate features and target
#     X = data_clean[features]
#     y = data_clean[target]

#     # Feature scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Train nearest neighbors model
#     n_neighbors = 5
#     nn = NearestNeighbors(n_neighbors=n_neighbors)
#     nn.fit(X_scaled)

#     # Streamlit app setup
#     st.title('SPAC and IPO Classifier')
#     st.sidebar.subheader('User Inputs')

#     input_features = ['revt', 'cshi', 'naicsh']
#     # User inputs
#     user_inputs = {}
#     for feature in input_features:
#         user_inputs[feature] = st.sidebar.number_input(f'Enter {feature}', value=X[feature].mean())

#     # Combine user inputs with default values for other features
#     full_input = {feature: X[feature].mean() for feature in features}
#     full_input.update(user_inputs)  # Update with user-provided values

#     # Prepare data for prediction
#     input_data = np.array([full_input[feature] for feature in features]).reshape(1, -1)
#     input_scaled = scaler.transform(input_data)
#     distances, indices = nn.kneighbors(input_scaled)
#     # # Predict and find nearest neighbors
#     # input_data = np.array([user_inputs[f] for f in features]).reshape(1, -1)
#     # input_scaled = scaler.transform(input_data)
#     # distances, indices = nn.kneighbors(input_scaled)

#     # Display nearest neighbors
#     st.write('### Results:')
#     st.write('Based on your inputs, your firm is most similar to the following firms:')
#     for i, idx in enumerate(indices[0]):
#         st.write(f"Firm {i + 1}: {master_merge.iloc[idx]['tic']}")
#     predicted_spac = 'SPAC' if y.iloc[indices[0]].mean() > 0.5 else 'not a SPAC'
#     st.write(f'Based on the characteristics of these firms, your firm is most likely {predicted_spac}')

#     # Plotting
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled)
#     fig, ax = plt.subplots()
#     scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5)
#     legend = ax.legend(*scatter.legend_elements(), title="Classes")
#     ax.add_artist(legend)
#     input_pca = pca.transform(input_scaled)
#     ax.scatter(input_pca[0, 0], input_pca[0, 1], c='red', s=100, label='Your Firm')
#     ax.set_title('PCA Plot of Firms with Nearest Neighbors')
#     ax.legend()
#     st.pyplot(fig)



# import streamlit as st
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import numpy as np

# # Load data
# merged_df = pd.read_csv('inputs/master_filtered_data.csv', low_memory=False)
# master_merge = pd.read_csv('inputs/masterMerge.csv')

# # Define all potential features
# all_potential_features = list(merged_df.columns.difference(['IS_SPAC']))  # Assuming 'IS_SPAC' is your only non-feature column
# target = 'IS_SPAC'

# # Optionally filter features based on data availability or relevance
# features = [col for col in all_potential_features if merged_df[col].notna().sum() > merged_df.shape[0] * 0.8]  # only use columns with less than 20% missing values

# # Clean data
# data_clean = merged_df.dropna(subset=features + [target])

# # Separate features and target
# X = data_clean[features]
# y = data_clean[target]

# # Feature scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train nearest neighbors model
# n_neighbors = 5
# nn = NearestNeighbors(n_neighbors=n_neighbors)
# nn.fit(X_scaled)

# # Streamlit app setup
# st.title('SPAC and IPO Classifier')
# st.sidebar.subheader('User Inputs')

# # Select a subset of features for user input
# input_features = ['revt', 'cshi', 'naicsh']  # Example subset of features for user inputs

# # User inputs
# user_inputs = {}
# for feature in input_features:
#     user_inputs[feature] = st.sidebar.number_input(f'Enter {feature}', value=X[feature].mean())

# # Default the rest of the features to their mean values
# default_inputs = {feature: X[feature].mean() for feature in features if feature not in input_features}
# user_inputs.update(default_inputs)  # Combine user inputs with default values

# # Prepare data for prediction
# input_data = np.array([user_inputs[feature] for feature in features]).reshape(1, -1)
# input_scaled = scaler.transform(input_data)
# distances, indices = nn.kneighbors(input_scaled)

# # Display nearest neighbors
# st.write('### Results:')
# st.write('Based on your inputs, your firm is most similar to the following firms:')
# for i, idx in enumerate(indices[0]):
#     st.write(f"Firm {i + 1}: {master_merge.iloc[idx]['tic']}")
# predicted_spac = 'SPAC' if y.iloc[indices[0]].mean() > 0.5 else 'not a SPAC'
# st.write(f'Based on the characteristics of these firms, your firm is most likely {predicted_spac}')
# st.write("Nearest Neighbors' Indicies:", indices)
# st.write("Nearest Neighbors' Features:")
# st.dataframe(master_merge.iloc[indices[0]])

# # For viewing the actual SPAC status of these neighbors
# st.write("Nearest Neighbors' SPAC Status:")
# st.write(y.iloc[indices[0]].values)

# # Plotting
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# fig, ax = plt.subplots()
# scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5)
# legend = ax.legend(*scatter.legend_elements(), title="Classes")
# ax.add_artist(legend)
# input_pca = pca.transform(input_scaled)
# ax.scatter(input_pca[0, 0], input_pca[0, 1], c='red', s=100, label='Your Firm')
# ax.set_title('PCA Plot of Firms with Nearest Neighbors')
# ax.legend()
# st.pyplot(fig)


