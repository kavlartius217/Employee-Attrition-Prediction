import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Function to train the model
def train_model(X, y):
    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical columns
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_columns)], remainder='passthrough')
    X_encoded = ct.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    model = KNeighborsClassifier()
    model.fit(X_scaled, y)
    return model, scaler, ct

# Function to make predictions
def predict_attrition(model, scaler, ct, new_data):
    # Encode new data
    new_data_encoded = ct.transform(new_data)
    new_data_scaled = scaler.transform(new_data_encoded)
    prediction = model.predict(new_data_scaled)
    return prediction[0]

def main():
    st.title('Employee Attrition Prediction')

    # Load the data
    data = load_data()

    # Separate features and target
    X = data.drop(['Attrition'], axis=1)
    y = data['Attrition']

    # Train the model
    model, scaler, ct = train_model(X, y)

    # User input for prediction
    st.sidebar.title('Make a Prediction')

    new_data = {}
    for column in X.columns:
        if X[column].dtype == 'object':
            new_data[column] = st.sidebar.selectbox(f'Select {column}', X[column].unique())
        else:
            new_data[column] = st.sidebar.number_input(f'Enter {column}')

    if st.sidebar.button('Predict'):
        new_data_df = pd.DataFrame([new_data])
        prediction = predict_attrition(model, scaler, ct, new_data_df)
        st.write(f"Predicted Attrition: {'Yes' if prediction == 1 else 'No'}")

if __name__ == "__main__":
    main()


