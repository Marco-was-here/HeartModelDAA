import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained model using pickle
with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Identify categorical and numerical columns for preprocessing
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# Function to preprocess user inputs
def preprocess_input(data):
    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ],
        remainder='passthrough'  # Handle columns not specified
    )
    
    # Transform the input data
    processed_data = preprocessor.fit_transform(data)
    return processed_data

# User input
st.title("Heart Disease Prediction")
st.write("Please input the following details to predict the likelihood of heart disease:")

# Define the input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Type 1", "Type 2", "Type 3", "Type 4"])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "Having ST-T wave abnormality", "Showing probable or definite left ventricular hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])


# Create a DataFrame from user input
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Preprocess the input data
processed_input = preprocess_input(input_data)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)
    st.write(f"Prediction: {'Heart Disease' if prediction[0] else 'No Heart Disease'}")
    st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
