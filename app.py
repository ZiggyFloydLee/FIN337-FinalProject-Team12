import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
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
master_merge = pd.read_csv('inputs/tickers_filtered.csv')

st.title('SPAC and IPO Classifier')
tab1, tab2 = st.tabs(["Nearest Neighbors Classifier", "Logistic Regression Plot"])

with st.expander('Why we choose this topic'):
    st.write("""Given the rise in Special Acquisition Companies within recent years, we recognized the need to identify private companies that aim to go public."
             """)
    
with st.expander("Data Description"):
    st.write("The data we used in this project is ")

with tab1:
    st.header('Nearest Neighbors Classifier')
    # Define features and target
    features = ['naicsh', 'caps', 'cshi', 'epspi', 'mkvalt']  # Ensure these column names exist in merged_df
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
    # st.title('SPAC and IPO Classifier')
    st.sidebar.subheader('User Inputs KNN Model')

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
    # st.write("Nearest Neighbors' Indicies:", indices)

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

with tab2:
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.header('Logistic Regression Plot')
    # Load and preprocess data
    df = pd.read_csv('inputs/masterMerge.csv')
    features = ['caps', 'epspi', 'naicsh', 'mkvalt', 'cshi']  # Expanded features
    target = 'IS_SPAC'

    # Preprocess data
    df[target] = df[target].fillna(0)
    df[target] = df[target].astype(int)
    imputer = SimpleImputer(strategy='mean')
    df[features] = imputer.fit_transform(df[features])
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Set up Streamlit interface
    st.title('Interactive Logistic Regression Plot')
    st.sidebar.header('User Inputs Logistic Regression Model')

    # User selects features for x and y axes
    x_feature = st.sidebar.selectbox('Select X Axis Feature', features, index=0)
    y_feature = st.sidebar.selectbox('Select Y Axis Feature', features, index=1)

    # Fit the logistic regression model on the selected features
    model = LogisticRegression()
    model.fit(df[[x_feature, y_feature]], df[target])  # Fit model only on two selected features

    # Generate a mesh grid for the contour plot
    grid_x, grid_y = np.meshgrid(np.linspace(df[x_feature].min(), df[x_feature].max(), 500),
                                np.linspace(df[y_feature].min(), df[y_feature].max(), 500))

    # Predict probabilities on the grid
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed
    contour = ax.contourf(grid_x, grid_y, probs, 25, cmap="RdBu", vmin=0, vmax=1)
    ax_c = fig.colorbar(contour)
    ax_c.set_label('P(SPAC)')
    # Plot scatter with the selected features
    ax.scatter(df[x_feature], df[y_feature], c=df[target], cmap="RdBu", edgecolor="white", lw=0.5)

    plt.title('Logistic Regression with Interactive User Inputs')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.gca().set_facecolor('white')

    # Adjust the y-axis limits to show higher y-values
    plt.ylim(np.min(grid_y), np.max(grid_y) * 10)  # Increase the upper limit by 10%

    # Set aspect to 'auto' or adjust it as needed
    plt.gca().set_aspect('auto')

    plt.tight_layout()
    st.pyplot(fig)
