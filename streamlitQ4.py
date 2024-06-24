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
            ('cat', OneHotEncoder(), categorical_cols)
        ])
    
    # Transform the input data
    processed_data = preprocessor.fit_transform(data)
    return processed_data

# User input
st.title("Heart Disease Prediction")
st.write("Please input the following details to predict the likelihood of heart disease:")
st.write("Marco Erlank | SMN45HWZ9")



age = st.number_input("Age", min_value=1, max_value=120)

# Sex: 1 = Male, 0 = Female
sex_options = {"Male": 1, "Female": 0}
sex = st.selectbox("Sex", list(sex_options.keys()))
sex_value = sex_options[sex]

# Chest Pain Type: 1, 2, 3, 4
cp_options = {f"Type {i}": i for i in range(1, 5)}
cp = st.selectbox("Chest Pain Type", list(cp_options.keys()))
cp_value = cp_options[cp]

trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
chol = st.number_input("Serum Cholesterol", min_value=100, max_value=600)

# Fasting Blood Sugar > 120 mg/dl: 1 = True, 0 = False
fbs_options = {"True": 1, "False": 0}
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(fbs_options.keys()))
fbs_value = fbs_options[fbs]

# Resting Electrocardiographic Results
restecg_options = {
    "Normal": 0,
    "Having ST-T wave abnormality": 1,
    "Showing probable or definite left ventricular hypertrophy": 2
}
restecg = st.selectbox("Resting Electrocardiographic Results", list(restecg_options.keys()))
restecg_value = restecg_options[restecg]

thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220)

# Exercise Induced Angina: 1 = Yes, 0 = No
exang_options = {"Yes": 1, "No": 0}
exang = st.selectbox("Exercise Induced Angina", list(exang_options.keys()))
exang_value = exang_options[exang]

oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0)

# Slope of the Peak Exercise ST Segment: 0 = Upsloping, 1 = Flat, 2 = Downsloping
slope_options = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = st.selectbox("Slope of the Peak Exercise ST Segment", list(slope_options.keys()))
slope_value = slope_options[slope]

ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3)

# Thalassemia: 0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect
thal_options = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}
thal = st.selectbox("Thalassemia", list(thal_options.keys()))
thal_value = thal_options[thal]

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_value],
    'cp': [cp_value],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs_value],
    'restecg': [restecg_value],
    'thalach': [thalach],
    'exang': [exang_value],
    'oldpeak': [oldpeak],
    'slope': [slope_value],
    'ca': [ca],
    'thal': [thal_value]
})

# Preprocess the input data
processed_input = preprocess_input(input_data)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)
    st.write(f"Prediction: {'Heart Disease' if prediction[0] else 'No Heart Disease'}")
    st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
